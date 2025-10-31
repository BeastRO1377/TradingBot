# bot_core.py
import asyncio
import logging
import json
import time
import os
import sys
import contextlib
import multiprocessing as mp
from collections import defaultdict, deque
import copy
from pathlib import Path
from typing import Dict, Any, Optional, Sequence
import datetime as dt
import pandas as pd
import numpy as np
import math
import random
import uuid
import collections
import inspect

from pybit.unified_trading import HTTP, WebSocket
from pybit.exceptions import InvalidRequestError
from requests.adapters import HTTPAdapter
from aiogram.enums import ParseMode
from ta_compat import ta

# Импортируем наши модули
import config
import utils
import strategies
import ai_ml
from telegram_bot import bot as telegram_bot
from signal_worker import start_worker_process
from data_manager import compute_supertrend
from utils import async_retry
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from websocket_monitor import get_monitor
import pickle

from trend_analyzer import detect_market_regime, adaptive_trailing_stop, summarize_regime
from intersession_strategies import is_quiet_market_window, generate_intersession_signals



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
        self.recently_closed_pnl_cache: Dict[str, Dict] = {}
        self.active_signals = set()
        self.strategy_cooldown_until: Dict[str, float] = {}
        self.last_entry_ts: Dict[str, float] = {}
        self._last_close_ts: Dict[str, float] = defaultdict(float)
        self.failed_orders: Dict[str, float] = {}
        self.reserve_orders: Dict[str, Dict] = {}
        self.closed_positions: Dict[str, Dict] = {}
        self.last_entry_comment: Dict[str, str] = {}
        self.pending_strategy_comments: Dict[str, str] = {}
        self.pending_open_exec: Dict[str, Dict[str, Any]] = {}
        self.pending_scale_ins: Dict[str, Dict[str, Any]] = {}
        self.momentum_cooldown_until = defaultdict(float)
        self.last_stop_price: Dict[str, float] = {}
        self.watch_tasks: Dict[str, asyncio.Task] = {}
        self.pending_signal_ids: Dict[str, str] = {}
        self.dom_signal_store: Dict[str, Dict[str, Any]] = {}
        self._last_aggressive_ts: Dict[str, float] = defaultdict(float)
        self._aggressive_tick_ts: Dict[str, float] = defaultdict(float)
        pattern_defaults = getattr(config, "PATTERN_SIGNAL_DEFAULTS", {})
        user_pattern_cfg = self.user_data.get("pattern_signal_settings") or {}
        merged_pattern_cfg = {**pattern_defaults, **user_pattern_cfg} if isinstance(user_pattern_cfg, dict) else pattern_defaults
        self.pattern_signals_enabled = bool(merged_pattern_cfg.get("ENABLED", True))
        self.pattern_signal_cooldown_sec = float(merged_pattern_cfg.get("COOLDOWN_SEC", 120.0))
        self.pattern_signal_min_confidence = float(merged_pattern_cfg.get("MIN_CONFIDENCE", 0.55))
        self._last_pattern_signal_ts: Dict[str, float] = defaultdict(float)
        self.user_state = getattr(self, "user_state", {})
        self.trailing_activated: Dict[str, bool] = {}
        self.trailing_activation_ts: Dict[str, float] = {} 
        self.last_trailing_update_ts = defaultdict(float)
        self.take_profit_price: Dict[str, float] = {}
        self.intersession_enabled = bool(
            user_data.get("intersession_enabled",
                           getattr(config, "INTERSESSION_TRADING_ENABLED", True))
        )

        self.last_sent_stop_price = {}       # последний отправленный нами SL (оптимистично)
        self.last_stop_attempt_ts = {}       # время последней попытки установки SL по символу
        # карта последних отправленных стопов (для RATCHET-защиты)

        # back-compat: чтобы старые места с last_known_stop_price не роняли процесс
        self.last_known_stop_price = self.last_sent_stop_price
        self._init_trailing_structs()

        self.aggressive_golden_cfg = dict(getattr(config, "AGGRESSIVE_GOLDEN_SETUP", {}))
        self.aggressive_golden_enabled = bool(self.aggressive_golden_cfg.get("ENABLED", True))
        self.aggressive_golden_tick_interval = float(self.aggressive_golden_cfg.get("TICK_INTERVAL_SEC", 2.0))

        self.wall_memory_lock = asyncio.Lock()


        self.position_ml_confidence = {}
        self.reentry_cooldown_sec = float(getattr(config, "REENTRY_COOLDOWN_SEC", 300.0))

        # Для антиспама логов/инфо по трейлингу
        if not hasattr(self, "_last_invalid_log_ts"):
            self._last_invalid_log_ts = {}    # символ -> ts
        if not hasattr(self, "_trailing_prev_stop"):
            self._trailing_prev_stop = {}     # символ -> последний залогированный SL
        if not hasattr(self, "_trailing_log_ts"):
            self._trailing_log_ts = {}        # символ -> ts последнего лога
        self.pending_position_meta: Dict[str, Dict[str, Any]] = {}

        self._intersession_last_signal_ts: Dict[str, float] = {}
        self._intersession_last_check_ts: Dict[str, float] = {}

        # анти-спам для SL-изменений (секунды)
        self.min_sl_retry_sec = float(self.user_data.get("MIN_SL_RETRY_SEC", 1.2))

        # Лучшие цены из стакана (L1) и кэш последней цены
        self.best_bid_map: Dict[str, float] = defaultdict(float)
        self.best_ask_map: Dict[str, float] = defaultdict(float)
        # на всякий случай актуализируем last
        self.last_price_map: Dict[str, float] = self.last_price_map if hasattr(self, "last_price_map") else defaultdict(float)
        # На всякий: убедимся, что symbol_meta/tick-карта есть
        self.symbol_meta: dict[str, dict] = getattr(self, "symbol_meta", {})
        self.price_tick_map: dict[str, float] = getattr(self, "price_tick_map", {})

        self.position_security_task: Optional[asyncio.Task] = None
        self.position_security_cfg: Dict[str, Any] = dict(config.POSITION_SECURITY)

        golden_defaults = dict(getattr(config, "GOLDEN_HUNTER", {}))
        user_golden_cfg = self.user_data.get("golden_settings") or {}
        combined_golden = {**golden_defaults, **user_golden_cfg}
        breakout_defaults = getattr(config, "BREAKOUT_HUNTER", {})
        combined_golden.setdefault("PRICE_DELTA_THRESHOLD", breakout_defaults.get("PRICE_DELTA_THRESHOLD", 0.2))
        combined_golden.setdefault("EXTREME_VOLUME_SPIKE", breakout_defaults.get("EXTREME_VOLUME_SPIKE", 75.0))
        combined_golden.setdefault("BREAKOUT_HUNT_WINDOW_SEC", breakout_defaults.get("WINDOW_SEC", 20))
        combined_golden.setdefault("BREAKOUT_CONFIRMATION_SCORE", breakout_defaults.get("REQUIRED_SCORE", 70))
        combined_golden.setdefault("REQUIRED_SCORE", breakout_defaults.get("REQUIRED_SCORE", 70))
        liquidity_defaults = getattr(config, "GOLDEN_LIQUIDITY_FILTER", {})
        combined_golden.setdefault("MIN_TURNOVER_24H", liquidity_defaults.get("MIN_TURNOVER_24H"))
        combined_golden.setdefault("MIN_AVG_VOL_1M", liquidity_defaults.get("MIN_AVG_VOL_1M"))
        self.golden_settings = combined_golden
        self.golden_tick_interval_sec = float(self.user_data.get("golden_tick_interval_sec", combined_golden.get("TICK_INTERVAL_SEC", 5.0)))
        self.golden_hunt_window_sec = float(combined_golden.get("HUNT_WINDOW_SEC", combined_golden.get("WINDOW_SEC", 20.0)))
        self.golden_required_score = float(combined_golden.get("REQUIRED_SCORE", 70.0))
        self.golden_price_delta_pct = float(combined_golden.get("PRICE_DELTA_PCT", 0.35))
        self.golden_volume_threshold = float(combined_golden.get("VOLUME_ANOMALY_MIN", 1.5))
        self.golden_price_weight = float(combined_golden.get("PRICE_WEIGHT", 40.0))
        self.golden_volume_weight = float(combined_golden.get("VOLUME_WEIGHT", 20.0))
        self.golden_flow_weight = float(combined_golden.get("FLOW_WEIGHT", 20.0))
        self.golden_flow_enabled = bool(combined_golden.get("FLOW_ENABLED", True))
        self.golden_min_adx = float(combined_golden.get("MIN_ADX", 0.0))
        self.golden_max_spread_ticks = int(combined_golden.get("MAX_SPREAD_TICKS", 6))
        self.golden_signal_cooldown_sec = float(combined_golden.get("SIGNAL_COOLDOWN_SEC", combined_golden.get("COOLDOWN_SEC", 120.0)))
        self.golden_hunter_cooldown_sec = float(combined_golden.get("COOLDOWN_SEC", 120.0))
        self.golden_min_turnover = utils.safe_to_float(combined_golden.get("MIN_TURNOVER_24H", 0.0))
        self.golden_min_avg_vol = utils.safe_to_float(combined_golden.get("MIN_AVG_VOL_1M", 0.0))
        self._golden_tick_ts: Dict[str, float] = defaultdict(float)
        self._golden_last_start_ts: Dict[str, float] = {}
        self.max_grid_orders_per_side = int(self.user_data.get("max_grid_orders_per_side", 3))

        strict_defaults = dict(getattr(config, "ML_STRICT_FILTERS", {}))
        user_strict_cfg = self.user_data.get("ML_STRICT_FILTERS") or {}
        combined_strict = {**strict_defaults, **user_strict_cfg}
        self.ml_strict_filters = combined_strict
        self.ml_strict_enabled = bool(combined_strict.get("ENABLED", True))
        self.ml_strict_min = utils.safe_to_float(combined_strict.get("MIN_WORKING_ML", 0.33))
        self.ml_strict_max = utils.safe_to_float(combined_strict.get("MAX_WORKING_ML", 0.85))

        pct1m_defaults = dict(getattr(config, "GOLDEN_PCT1M_FILTER", {}))
        self.golden_pct1m_filter = pct1m_defaults
        cvd_defaults = dict(getattr(config, "GOLDEN_CVD_FILTER", {}))
        self.golden_cvd_filter = cvd_defaults
        adx_defaults = dict(getattr(config, "GOLDEN_ADX_FILTER", {}))
        self.golden_adx_filter = adx_defaults
        price_exhaust_defaults = dict(getattr(config, "GOLDEN_PRICE_EXHAUSTION_FILTER", {}))
        self.golden_price_exhaustion_filter = price_exhaust_defaults
        signal_defaults = dict(getattr(config, "SIGNAL_MODEL", {}))
        self.signal_model_cfg = signal_defaults
        self.signal_model_enabled = bool(signal_defaults.get("ENABLED", True))
        self.signal_model_threshold = utils.safe_to_float(signal_defaults.get("THRESHOLD", 0.6))
        self.signal_prev_state: Dict[str, Dict[str, float]] = defaultdict(dict)

        self.position_mode = 0
        self.flea_positions_count = 0
        self.flea_cooldown_until: Dict[str, float] = {}
        self.leverage = utils.safe_to_float(user_data.get("leverage", 10.0))
        self.qty_step_map: Dict[str, float] = {}
        self.max_qty_map: Dict[str, float] = {}
        self.min_qty_map: Dict[str, float] = {}
        #self.price_tick_map: Dict[str, float] = {}
        self.ml_inferencer: Optional[ai_ml.MLXInferencer] = None
        self.training_data = deque(maxlen=5000)
        self.policy_feature_dim = len(config.FEATURE_KEYS) + 1
        try:
            self.ml_policy = ai_ml.OnlinePolicyLearner(self.policy_feature_dim, getattr(config, "ML_POLICY", {}))
        except Exception as exc:
            logger.error(f"[PolicyInit] Не удалось инициализировать ML-политику: {exc}", exc_info=True)
            self.ml_policy = None
        self.pending_policy_vectors: Dict[str, Dict[str, Any]] = {}
        self.pending_policy_records: Dict[str, Dict[str, Any]] = {}
        self.learning_positions: Dict[str, Dict[str, Any]] = {}
        self.ai_circuit_open_until = 0.0
        self._ai_inflight_signals = set()
        self.apply_user_settings()
        self.momentum_cooldown_until = defaultdict(float)
        self.session = HTTP(
            testnet=False, demo=(self.mode == "demo"),
            api_key=self.api_key, api_secret=self.api_secret, timeout=30
        )
        self._sanitize_http_headers(self.session)
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
        self.leverage = 10.0
        self.feature_extraction_sem = asyncio.Semaphore(4)
        # Кэш для market features (TTL 5 секунд)
        self._feature_cache = {}
        self._feature_cache_ttl = 5.0
        self.averaging_orders_count: Dict[str, int] = defaultdict(int)
        self.last_hf_check_ts = defaultdict(float)
        self._wall_memory_file = config.WALL_MEMORY_FILE
        self.position_peak_price: Dict[str, float] = {}
        self.dom_wall_memory = {}
        self.wall_watch_list = {}
        self._wall_memory_save_task = None
        self.trailing_lock = defaultdict(asyncio.Lock)
        self.dom_dataset_lock = asyncio.Lock()
        self._orderbook_metrics = defaultdict(lambda: {"bid_depth": 0.0, "ask_depth": 0.0, "depth_ratio": 0.0, "orderbook_imbalance": 0.0})
        self._fail_fast_tasks: Dict[str, asyncio.Task] = {}

    def _sanitize_http_headers(self, http_client):
        client = getattr(http_client, "client", None)
        if client is None:
            return
        sanitized = {}
        for key, value in client.headers.items():
            if isinstance(value, str):
                try:
                    value.encode("latin-1")
                    sanitized[key] = value
                except UnicodeEncodeError:
                    safe = value.encode("ascii", "ignore").decode("ascii")
                    sanitized[key] = safe or "pybit"
            else:
                sanitized[key] = value
        client.headers.clear()
        client.headers.update(sanitized)

    def apply_user_settings(self):
        cfg = self.user_data
        logger.info("Применение новых настроек из user_state.json...")
        self._wall_memory_file = config.WALL_MEMORY_FILE
        security_cfg = dict(config.POSITION_SECURITY)
        user_security = cfg.get("position_security")
        if isinstance(user_security, dict):
            for key, value in user_security.items():
                if value is not None:
                    security_cfg[key] = value
        self.position_security_cfg = security_cfg
        self.strategy_mode = cfg.get("strategy_mode", "full")
        self.ai_primary_model = cfg.get("ai_primary_model", config.AI_PRIMARY_MODEL)
        self.ai_advisor_model = cfg.get("ai_advisor_model", config.AI_ADVISOR_MODEL)
        self.ollama_primary_openai = cfg.get("ollama_primary_openai", config.OLLAMA_PRIMARY_OPENAI)
        self.ollama_advisor_openai = cfg.get("ollama_advisor_openai", config.OLLAMA_ADVISOR_OPENAI)
        self.ai_timeout_sec = float(cfg.get("ai_timeout_sec", 60.0))
        self.POSITION_VOLUME = utils.safe_to_float(cfg.get("volume", 1000.0))
        self.MAX_TOTAL_VOLUME = utils.safe_to_float(cfg.get("max_total_volume", 5000.0))
        self.leverage = utils.safe_to_float(cfg.get("leverage", 10.0))
        self.entry_cooldown_sec = int(cfg.get("entry_cooldown_sec", 30))
        self.tactical_entry_window_sec = int(cfg.get("tactical_entry_window_sec", 300))
        self.squeeze_ai_confirm_interval_sec = float(cfg.get("squeeze_ai_confirm_interval_sec", 2.0))
        self.ai_advice_interval_min = int(cfg.get("AI_ADVICE_INTERVAL_MIN", 5))
        self.listing_age_min = int(cfg.get("listing_age_min_minutes", 1440))
        self.intersession_enabled = bool(
            cfg.get(
                "intersession_enabled",
                getattr(config, "INTERSESSION_TRADING_ENABLED", self.intersession_enabled),
            )
        )
        golden_defaults = dict(getattr(config, "GOLDEN_HUNTER", {}))
        user_golden_cfg = cfg.get("golden_settings") or {}
        combined_golden = {**golden_defaults, **user_golden_cfg}
        breakout_defaults = getattr(config, "BREAKOUT_HUNTER", {})
        combined_golden.setdefault("PRICE_DELTA_THRESHOLD", breakout_defaults.get("PRICE_DELTA_THRESHOLD", 0.2))
        combined_golden.setdefault("EXTREME_VOLUME_SPIKE", breakout_defaults.get("EXTREME_VOLUME_SPIKE", 75.0))
        combined_golden.setdefault("BREAKOUT_HUNT_WINDOW_SEC", breakout_defaults.get("WINDOW_SEC", 20))
        combined_golden.setdefault("BREAKOUT_CONFIRMATION_SCORE", breakout_defaults.get("REQUIRED_SCORE", 70))
        combined_golden.setdefault("REQUIRED_SCORE", breakout_defaults.get("REQUIRED_SCORE", 70))
        liquidity_defaults = getattr(config, "GOLDEN_LIQUIDITY_FILTER", {})
        combined_golden.setdefault("MIN_TURNOVER_24H", liquidity_defaults.get("MIN_TURNOVER_24H"))
        combined_golden.setdefault("MIN_AVG_VOL_1M", liquidity_defaults.get("MIN_AVG_VOL_1M"))
        self.golden_settings = combined_golden
        self.golden_tick_interval_sec = float(cfg.get("golden_tick_interval_sec", combined_golden.get("TICK_INTERVAL_SEC", self.golden_tick_interval_sec)))
        self.golden_hunt_window_sec = float(combined_golden.get("HUNT_WINDOW_SEC", combined_golden.get("WINDOW_SEC", self.golden_hunt_window_sec)))
        self.golden_required_score = float(combined_golden.get("REQUIRED_SCORE", self.golden_required_score))
        self.golden_price_delta_pct = float(combined_golden.get("PRICE_DELTA_PCT", self.golden_price_delta_pct))
        self.golden_volume_threshold = float(combined_golden.get("VOLUME_ANOMALY_MIN", self.golden_volume_threshold))
        self.golden_price_weight = float(combined_golden.get("PRICE_WEIGHT", self.golden_price_weight))
        self.golden_volume_weight = float(combined_golden.get("VOLUME_WEIGHT", self.golden_volume_weight))
        self.golden_flow_weight = float(combined_golden.get("FLOW_WEIGHT", self.golden_flow_weight))
        self.golden_flow_enabled = bool(combined_golden.get("FLOW_ENABLED", self.golden_flow_enabled))
        self.golden_min_adx = float(combined_golden.get("MIN_ADX", self.golden_min_adx))
        self.golden_max_spread_ticks = int(combined_golden.get("MAX_SPREAD_TICKS", self.golden_max_spread_ticks))
        self.golden_signal_cooldown_sec = float(combined_golden.get("SIGNAL_COOLDOWN_SEC", self.golden_signal_cooldown_sec))
        self.golden_hunter_cooldown_sec = float(combined_golden.get("COOLDOWN_SEC", self.golden_hunter_cooldown_sec))
        base_turnover = getattr(self, "golden_min_turnover", 0.0)
        base_avg_vol = getattr(self, "golden_min_avg_vol", 0.0)
        self.golden_min_turnover = utils.safe_to_float(combined_golden.get("MIN_TURNOVER_24H", base_turnover))
        self.golden_min_avg_vol = utils.safe_to_float(combined_golden.get("MIN_AVG_VOL_1M", base_avg_vol))
        self.max_grid_orders_per_side = int(cfg.get("max_grid_orders_per_side", self.max_grid_orders_per_side))

        aggressive_defaults = dict(getattr(config, "AGGRESSIVE_GOLDEN_SETUP", {}))
        user_aggressive_cfg = cfg.get("aggressive_golden_settings") or {}
        if isinstance(user_aggressive_cfg, dict):
            combined_aggressive = {**aggressive_defaults, **user_aggressive_cfg}
        else:
            combined_aggressive = aggressive_defaults
        self.aggressive_golden_cfg = combined_aggressive
        self.aggressive_golden_enabled = bool(combined_aggressive.get("ENABLED", True))
        self.aggressive_golden_tick_interval = float(combined_aggressive.get("TICK_INTERVAL_SEC", 2.0))

        strict_defaults = dict(getattr(config, "ML_STRICT_FILTERS", {}))
        user_strict_cfg = cfg.get("ML_STRICT_FILTERS") or {}
        combined_strict = {**strict_defaults, **user_strict_cfg}
        self.ml_strict_filters = combined_strict
        self.ml_strict_enabled = bool(combined_strict.get("ENABLED", self.ml_strict_enabled))
        self.ml_strict_min = utils.safe_to_float(combined_strict.get("MIN_WORKING_ML", self.ml_strict_min))
        self.ml_strict_max = utils.safe_to_float(combined_strict.get("MAX_WORKING_ML", self.ml_strict_max))

        pct1m_defaults = dict(getattr(config, "GOLDEN_PCT1M_FILTER", {}))
        user_pct1m_cfg = cfg.get("GOLDEN_PCT1M_FILTER") or {}
        self.golden_pct1m_filter = {**pct1m_defaults, **user_pct1m_cfg}

        cvd_defaults = dict(getattr(config, "GOLDEN_CVD_FILTER", {}))
        user_cvd_cfg = cfg.get("GOLDEN_CVD_FILTER") or {}
        self.golden_cvd_filter = {**cvd_defaults, **user_cvd_cfg}

        adx_defaults = dict(getattr(config, "GOLDEN_ADX_FILTER", {}))
        user_adx_cfg = cfg.get("GOLDEN_ADX_FILTER") or {}
        self.golden_adx_filter = {**adx_defaults, **user_adx_cfg}

        price_exhaust_defaults = dict(getattr(config, "GOLDEN_PRICE_EXHAUSTION_FILTER", {}))
        user_price_exhaust_cfg = cfg.get("GOLDEN_PRICE_EXHAUSTION_FILTER") or {}
        self.golden_price_exhaustion_filter = {**price_exhaust_defaults, **user_price_exhaust_cfg}

        signal_defaults = dict(getattr(config, "SIGNAL_MODEL", {}))
        user_signal_cfg = cfg.get("SIGNAL_MODEL") or {}
        merged_signal = {**signal_defaults, **user_signal_cfg}
        self.signal_model_cfg = merged_signal
        self.signal_model_enabled = bool(merged_signal.get("ENABLED", self.signal_model_enabled))
        self.signal_model_threshold = utils.safe_to_float(merged_signal.get("THRESHOLD", self.signal_model_threshold))

        self.reentry_cooldown_sec = float(cfg.get("reentry_cooldown_sec", getattr(config, "REENTRY_COOLDOWN_SEC", self.reentry_cooldown_sec)))
        logger.info(
            f"Настройки для пользователя {self.user_id} применены: "
            f"Режим='{self.strategy_mode}', "
            f"Объем={self.POSITION_VOLUME}, "
            f"Макс.Объем={self.MAX_TOTAL_VOLUME}, "
            f"Интервал AI={self.ai_advice_interval_min} мин."
        )


    def on_config_reload(self):
        logger.info("Перезагрузка config для пользователя %s", self.user_id)
        self._wall_memory_file = config.WALL_MEMORY_FILE
        try:
            self.apply_user_settings()
        except Exception as e:
            logger.error("Не удалось применить настройки пользователя после reload config: %s", e, exc_info=True)
        if self.ml_inferencer is not None:
            try:
                self.ml_inferencer = ai_ml.MLXInferencer(
                    model_path=config.ML_MODEL_PATH,
                    scaler_path=config.SCALER_PATH,
                )
            except Exception as e:
                logger.error("Не удалось обновить ML-инференсер после reload config: %s", e, exc_info=True)


    def _get_recent_ohlcv_1m(self, symbol: str, lookback: int = 60) -> Optional[pd.DataFrame]:
        """Берём последние минутные свечи из self.shared_ws.candles_data и приводим к open/high/low/close/volume."""
        try:
            raw = list(self.shared_ws.candles_data.get(symbol, []))[-lookback:]
            if not raw:
                return None
            df = pd.DataFrame(raw)
            # Жёсткое приведение имён
            def pick(col, alts):
                return pd.to_numeric(df.get(col, df.get(alts[0], df.get(alts[1], np.nan))), errors="coerce")
            df["open"]   = pick("open",   ["openPrice","o"])
            df["high"]   = pick("high",   ["highPrice","h"])
            df["low"]    = pick("low",    ["lowPrice","l"])
            df["close"]  = pick("close",  ["closePrice","c"])
            df["volume"] = pd.to_numeric(df.get("volume", df.get("turnover")), errors="coerce").fillna(0.0)
            df = df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)
            return df if len(df) >= max(15, lookback//2) else None
        except Exception:
            logger.exception(f"[{symbol}] _get_recent_ohlcv_1m failed")
            return None


    async def _place_grid_orders(self, symbol: str, buys: list[float], sells: list[float], last_price: float):
        """
        Ставит симметричную сетку лимитных заявок (GTC). Ставим один раз; флаг хранится в reserve_orders.
        """
        try:
            if symbol in self.open_positions or symbol in self.pending_orders:
                return
            grid_state = self.reserve_orders.get(symbol, {})
            if grid_state.get("grid_active"):
                return

            await self.ensure_symbol_meta(symbol)
            step = float(self.qty_step_map.get(symbol, 0.001) or 0.001)
            min_qty = float(self.min_qty_map.get(symbol, step) or step)
            tick = float(self.price_tick_map.get(symbol, 1e-6) or 1e-6)

            levels = buys + sells
            if not levels:
                return

            volume = float(self.POSITION_VOLUME)
            per_level_qty = (volume / max(last_price, 1e-9)) / max(len(levels),1)
            per_level_qty = math.floor(max(min_qty, per_level_qty) / step) * step
            if per_level_qty <= 0:
                return

            for p in buys:
                price = math.floor(p / tick) * tick
                await self.place_unified_order(symbol, "Buy", per_level_qty, "Limit", price=price, comment="INTRAQUIET_GRID_BUY")
            for p in sells:
                price = math.ceil(p / tick) * tick
                await self.place_unified_order(symbol, "Sell", per_level_qty, "Limit", price=price, comment="INTRAQUIET_GRID_SELL")

            self.reserve_orders.setdefault(symbol, {})["grid_active"] = True
            logger.info(f"[{symbol}] GRID placed: {len(buys)} buys / {len(sells)} sells, qty≈{per_level_qty}")
        except Exception as e:
            logger.error(f"[{symbol}] _place_grid_orders failed: {e}", exc_info=True)



    def _init_trailing_structs(self) -> None:
        # тайминги обновлений и логирования
        self.last_trailing_update_ts = getattr(self, "last_trailing_update_ts", {})
        self._trailing_log_ts = getattr(self, "_trailing_log_ts", {})
        self._trailing_prev_stop = getattr(self, "_trailing_prev_stop", {})

        # блокировки по символу
        if not hasattr(self, "trailing_lock") or not isinstance(self.trailing_lock, dict):
            self.trailing_lock = {}
        # лениво создаём locks в _ensure_trailing_state

        # отправленные стопы (для ratchet)
        self.last_sent_stop_price = getattr(self, "last_sent_stop_price", {})

        # котировки лучшего бида/аска (если не ведёшь — оставим пустыми)
        self.best_bid_map = getattr(self, "best_bid_map", {})
        self.best_ask_map = getattr(self, "best_ask_map", {})

        # история ADX для расчёта наклона
        self._adx_hist = getattr(self, "_adx_hist", {})

        # таймер «застревания» у стены
        self._wall_stall_since = getattr(self, "_wall_stall_since", {})


    async def get_dom_next_wall_price(self, symbol: str, side: str) -> float | None:
        """
        Верни float цену ближайшей стены в сторону сделки, либо None если данных нет.
        Если у тебя в DOM модуле есть другой метод - используй его и удали эту заглушку.
        """
        try:
            # пример как это может выглядеть, подмени на свою реализацию:
            dom = getattr(self, "last_dom_snapshot", {}).get(symbol) or {}
            if side == "buy":
                return float(dom.get("next_ask_wall_price") or 0) or None
            else:
                return float(dom.get("next_bid_wall_price") or 0) or None
        except Exception:
            return None


    async def _handle_realtime_price_tick(self, symbol: str, price: float):
        """
        Вызывается при каждом обновлении цены по тикеру через WebSocket.
        Реализует реактивное обновление трейлинга.
        """
        pos = self.open_positions.get(symbol)
        if not pos:
            return

        trailing_mode = pos.get("trailing_mode", "simple_gap")
        if trailing_mode == "simple_gap":
            await self._run_simple_gap_trailing(symbol, price, pos)
        elif trailing_mode == "dynamic":
            await self._run_dynamic_atr_trailing(symbol, price, pos)
        elif trailing_mode == "adaptive":
            await self._run_adaptive_trailing(symbol, price, pos)


    def _ensure_trailing_state(self, symbol: str) -> None:
        # lock на символ
        if symbol not in self.trailing_lock:
            self.trailing_lock[symbol] = asyncio.Lock()

        # тайминги последнего апдейта
        if symbol not in self.last_trailing_update_ts:
            self.last_trailing_update_ts[symbol] = 0.0

        # последний отправленный SL (для ratchet)
        if symbol not in self.last_sent_stop_price:
            self.last_sent_stop_price[symbol] = 0.0

        # логи/предыдущий кандидат
        if symbol not in self._trailing_log_ts:
            self._trailing_log_ts[symbol] = 0.0
        if symbol not in self._trailing_prev_stop:
            self._trailing_prev_stop[symbol] = None

        # история ADX
        if symbol not in self._adx_hist:
            self._adx_hist[symbol] = collections.deque(maxlen=int(
                config.TRAILING_MODES.get("dynamic", {}).get("ADX_SLOPE_WINDOW", 6)
            ))

        # таймер «застревания» у стены
        if symbol not in self._wall_stall_since:
            self._wall_stall_since[symbol] = None


    async def _get_opposite_wall_price(self, symbol: str, side: str, last_price: float) -> float | None:
        opp = "sell" if (side or "").lower() == "buy" else "buy"
        getter = getattr(self, "get_dom_next_wall_price", None)
        if not callable(getter):
            return None
        try:
            maybe = getter(symbol, opp)
            price = await maybe if inspect.iscoroutine(maybe) else maybe
            price_f = utils.safe_to_float(price or 0.0)
            return price_f if price_f > 0 else None
        except Exception:
            return None


    def _ensure_l2_maps(self) -> None:
        if not hasattr(self, "best_bid_map"):
            self.best_bid_map = {}
        if not hasattr(self, "best_ask_map"):
            self.best_ask_map = {}
        if not hasattr(self, "last_price_map"):
            self.last_price_map = {}

    def _get_tick(self, symbol: str) -> float:
        # tick из meta → из карты → безопасный минимум
        return utils.safe_to_float(
            ((self.symbol_meta.get(symbol, {}) or {}).get("priceFilter", {}) or {}
            ).get("tickSize", self.price_tick_map.get(symbol, 0.0) or 1e-6)
        )

    def get_best_bid(self, symbol: str, default_last: float | None = None) -> float:
        self._ensure_l2_maps()
        bid = utils.safe_to_float(self.best_bid_map.get(symbol, 0.0))
        if bid > 0:
            return bid
        # фолбэк к last ± tick, если стакан ещё не успел прийти
        last = utils.safe_to_float(
            self.last_price_map.get(symbol, default_last if default_last is not None else 0.0)
        )
        tick = self._get_tick(symbol)
        return last - tick if last > 0 else 0.0

    def get_best_ask(self, symbol: str, default_last: float | None = None) -> float:
        self._ensure_l2_maps()
        ask = utils.safe_to_float(self.best_ask_map.get(symbol, 0.0))
        if ask > 0:
            return ask
        last = utils.safe_to_float(
            self.last_price_map.get(symbol, default_last if default_last is not None else 0.0)
        )
        tick = self._get_tick(symbol)
        return last + tick if last > 0 else 0.0


    def safe_last_price(self, symbol: str) -> float:
        """
        Единое безопасное получение last_price.
        Проверяем все варианты ключей, которые могли «гулять».
        """
        td = (getattr(self.shared_ws, "ticker_data", {}) or {}).get(symbol, {}) or {}
        lp = td.get("last_price") or td.get("lastPrice") or td.get("markPrice") or td.get("indexPrice")
        return utils.safe_to_float(lp)

    def _get_symbol_turnover(self, symbol: str) -> float:
        ticker = (getattr(self.shared_ws, "ticker_data", {}) or {}).get(symbol, {}) or {}
        for key in (
            "turnover24h",
            "turnover24hUsd",
            "turnover24hUSDT",
            "turnover24hQuote",
            "turnover24hBase",
            "turnover24",
        ):
            val = utils.safe_to_float(ticker.get(key))
            if val:
                return val
        return utils.safe_to_float(ticker.get("volume24h") or ticker.get("volume24H"))

    def _determine_position_volume(self, symbol: str, base_volume: float, features: dict | None = None) -> float:
        """
        Определяет объём входа c учётом ML-уверенности и прочих факторов.
        ГАРАНТИЯ: всегда возвращает неотрицательный float и не падает по UnboundLocalError.
        """
        # --- База: всегда инициализируем adjusted ---
        try:
            adjusted = float(base_volume)
        except Exception:
            adjusted = float(getattr(self, "POSITION_VOLUME", 0.0))
        if not np.isfinite(adjusted) or adjusted <= 0:
            adjusted = float(getattr(self, "POSITION_VOLUME", 0.0))

        # --- Фичи могут быть None ---
        features = features or {}

        # --- ML-мультипликатор сайзинга ---
        # хотим диапазон ~[0.8, 1.5], причём при ml_conf=0.5 → 1.0
        ml_conf = 0.5
        try:
            # сначала ищем в features, затем в pending meta/candidate (если кто-то прокинул раньше)
            ml_conf = float(features.get("ml_confidence", features.get("ml_conf", 0.5)))
            if not np.isfinite(ml_conf):
                ml_conf = 0.5
            ml_conf = max(0.0, min(1.0, ml_conf))  # клип 0..1
        except Exception:
            ml_conf = 0.5

        if ml_conf >= 0.5:
            ml_mul = 1.0 + 1.0 * (ml_conf - 0.5)  # 0.5→1.0, 1.0→1.5
        else:
            ml_mul = 1.0 - 0.4 * (0.5 - ml_conf)  # 0.0→0.8, 0.5→1.0

        ml_mul = max(0.8, min(1.5, ml_mul))
        adjusted *= ml_mul

        # --- (опционально) твои существующие множители/ограничители ниже ---
        # пример: лимит на символ
        max_per_symbol = float(getattr(config, "MAX_POSITION_VOLUME_PER_SYMBOL", 0.0) or 0.0)
        if max_per_symbol > 0:
            adjusted = min(adjusted, max_per_symbol)

        # пример: анти-пыль (не меньше минимального объёма в $)
        min_notional = float(getattr(config, "MIN_NOTIONAL_USD", 0.0) or 0.0)
        if min_notional > 0:
            # если есть last price в features, можно прикинуть нижнюю границу по qty
            last_price = features.get("last_price") or features.get("price") or 0.0
            try:
                last_price = float(last_price)
            except Exception:
                last_price = 0.0
            if last_price > 0:
                adjusted = max(adjusted, min_notional)

        # клип финального результата (страховка от NaN/Inf/отрицательных)
        if not np.isfinite(adjusted) or adjusted < 0:
            adjusted = float(getattr(self, "POSITION_VOLUME", 0.0))

        return float(adjusted)


    def _round_to_tick(self, symbol: str, price: float, side: str) -> float:
        """
        [ИСПРАВЛЕННАЯ ВЕРСИЯ] Корректно округляет цену стоп-лосса:
        - ВНИЗ для лонг-позиций (Buy).
        - ВВЕРХ для шорт-позиций (Sell).
        """
        tick = float(self.price_tick_map.get(symbol, 1e-6) or 1e-6)
        if tick <= 0:
            return float(price)
        
        # Для стопа на покупку (лонг) цена должна быть НИЖЕ, округляем ВНИЗ.
        if side.lower() == "buy":
            rounded = math.floor(price / tick) * tick
        # Для стопа на продажу (шорт) цена должна быть ВЫШЕ, округляем ВВЕРХ.
        else:
            rounded = math.ceil(price / tick) * tick
            
        return float(f"{rounded:.10f}")

    def _round_to_step(self, symbol: str, qty: float, *, mode: str = "floor") -> Decimal:
        """
        Приводит количество к шагу биржи. Использует Decimal, чтобы избежать артефактов
        вида 0.30000000000001, и поддерживает режимы 'floor' и 'ceil'.
        """
        step = utils.safe_to_float(self.qty_step_map.get(symbol, 0.0), 0.0)
        if step <= 0:
            return Decimal(str(qty))
        step_str = f"{step:.16f}".rstrip("0").rstrip(".")
        if not step_str:
            step_str = "1"
        try:
            d_step = Decimal(step_str)
            d_qty = Decimal(str(qty))
            units = d_qty / d_step
            rounding = ROUND_DOWN if mode != "ceil" else ROUND_UP
            units = units.quantize(Decimal("1"), rounding=rounding)
            rounded = (units * d_step).quantize(d_step.normalize())
            return rounded
        except Exception:
            # fallback на float-округление
            if mode == "ceil":
                rounded = math.ceil(qty / step) * step
            else:
                rounded = math.floor(qty / step) * step
            return Decimal(str(float(f"{rounded:.10f}")))

    def _best_mid_from_orderbook(self, symbol: str) -> float:
        """
        Быстрый фолбэк: mid из лучшего бида/аска. Если есть только одна сторона — берём её.
        """
        try:
            ob = getattr(self.shared_ws, "orderbooks", {}).get(symbol) or {}
            bids = ob.get("bids", {}) or {}
            asks = ob.get("asks", {}) or {}
            if not bids and not asks:
                return 0.0

            best_bid = max(bids.keys()) if bids else None
            best_ask = min(asks.keys()) if asks else None

            if best_bid is not None and best_ask is not None:
                return (float(best_bid) + float(best_ask)) / 2.0
            if best_bid is not None:
                return float(best_bid)
            if best_ask is not None:
                return float(best_ask)
        except Exception:
            pass
        return 0.0


    async def _resolve_last_price(self, symbol: str, features: dict | None = None) -> float:
        """
        Берём last/mark из:
        1) переданных features,
        2) shared_ws.ticker_data,
        3) mid из ордербука (фолбэк).
        """
        # 1) features
        lp = utils.safe_to_float((features or {}).get("lastPrice") or (features or {}).get("markPrice") or 0.0)
        if lp and lp > 0:
            return float(lp)

        # 2) глобальный тикер
        t = getattr(self.shared_ws, "ticker_data", {}).get(symbol) or {}
        lp = utils.safe_to_float(t.get("lastPrice") or t.get("markPrice") or 0.0)
        if lp and lp > 0:
            return float(lp)

        # 3) mid из ордербука
        return float(self._best_mid_from_orderbook(symbol))


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

    async def on_ready(self):
        logger.info(f"Бот {self.user_id} запущен")
        if hasattr(strategies, 'init_bot_memory'):
            strategies.init_bot_memory(self)
        else:
            if not hasattr(self, 'dom_wall_memory'):
                self.dom_wall_memory = {}
            if not hasattr(self, 'wall_watch_list'):
                self.wall_watch_list = {}

    # --- ИСПРАВЛЕННАЯ ВЕРСИЯ ---
    # Сохранена ваша улучшенная версия с двумя независимыми сканерами.
    # Это грамотное решение для разделения ВЧ и НЧ-логики.
    async def start(self):        
        await self._sync_server_time()
        self._load_wall_memory()
        strategies.init_mlx_components(self)
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
        asyncio.create_task(self._cleanup_pnl_cache())
        asyncio.create_task(self._cleanup_feature_cache()) 
        asyncio.create_task(self.reload_settings_loop())

        if not self._wall_memory_save_task or self._wall_memory_save_task.done():
            self._wall_memory_save_task = asyncio.create_task(self._wall_memory_save_loop())

        await self.update_open_positions()
        await self._cache_all_symbol_meta()
        await self.secure_open_positions(reason="Boot auto-secure")
        await self.setup_private_ws()
        
        # Запускаем два независимых сканера: один для Golden Setup, другой для ВЧ-стратегий
        asyncio.create_task(self._scanner_worker_loop()) # ВЧ-Воркер
        if self.position_security_cfg.get("ENABLED", True):
            self.position_security_task = asyncio.create_task(self._position_security_loop())
        
        logger.info(f"Бот для пользователя {self.user_id} полностью готов к работе.")

    async def _scanner_worker_loop(self, interval_sec: float = 1.0):
        """
        Фоновый воркер, который периодически сканирует все монеты в случайном порядке
        для поиска высокочастотных сигналов.
        """
        await self.shared_ws.ready_event.wait()
        logger.info(f"Высокочастотный сканер запущен для пользователя {self.user_id}.")
        while True:
            try:
                watchlist = list(self.shared_ws.watchlist)
                if not watchlist:
                    await asyncio.sleep(interval_sec)
                    continue

                # Перемешиваем список для случайного порядка сканирования
                random.shuffle(watchlist)
                
                # Создаем задачи для проверки каждой монеты с ограничением concurrency
                semaphore = asyncio.Semaphore(20)  # Максимум 20 одновременных задач
                async def limited_dispatcher(symbol):
                    async with semaphore:
                        return await strategies.high_frequency_dispatcher(self, symbol)
                
                tasks = [limited_dispatcher(symbol) for symbol in watchlist]
                await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                logger.error(f"Ошибка в цикле ВЧ-сканера: {e}", exc_info=True)
            
            # Ждем перед следующим полным сканированием
            await asyncio.sleep(interval_sec)


    async def train_and_save_model(self):
        """
        Загружает историю сделок, подготавливает данные и обучает ML-модель.
        """
        logger.info("Начинается процесс сбора данных и обучения модели...")
        MIN_SAMPLES_FOR_TRAINING = 100

        try:
            # 1. Загрузка данных
            if not config.TRADES_UNIFIED_CSV_PATH.exists():
                logger.error(f"Файл с историей сделок не найден: {config.TRADES_UNIFIED_CSV_PATH}. Обучение невозможно.")
                return

            df = pd.read_csv(config.TRADES_UNIFIED_CSV_PATH)
            logger.info(f"Загружено {len(df)} записей из истории сделок.")

            # 2. Подготовка данных
            df.dropna(subset=['pnl_pct'], inplace=True)
            
            if len(df) < MIN_SAMPLES_FOR_TRAINING:
                logger.error(f"Недостаточно данных для обучения. Найдено {len(df)} закрытых сделок, требуется минимум {MIN_SAMPLES_FOR_TRAINING}.")
                return

            logger.info(f"Подготовка {len(df)} сэмплов для обучения...")
            training_data = []
            for _, row in df.iterrows():
                features_vector = []
                for key in config.FEATURE_KEYS:
                    value = row.get(key, 0.0)
                    features_vector.append(utils.safe_to_float(value, default=0.0))
                
                target = utils.safe_to_float(row['pnl_pct'])

                training_data.append({
                    "features": features_vector,
                    "target": target
                })

            logger.info("Данные подготовлены. Запуск обучения модели MLX...")

            # 3. Обучение модели
            model, scaler = await asyncio.to_thread(
                ai_ml.train_golden_model_mlx, training_data
            )
            
            if model and scaler:
                logger.info("Обучение успешно завершено.")
                
                # 4. Сохранение модели и скейлера
                await asyncio.to_thread(
                    ai_ml.save_mlx_checkpoint, model, scaler
                )
                logger.info(f"Модель и скейлер сохранены в {config.ML_MODEL_PATH} и {config.SCALER_PATH}")
            else:
                logger.error("Функция обучения не вернула модель или скейлер.")

        except Exception as e:
            logger.critical(f"Критическая ошибка в процессе обучения модели: {e}", exc_info=True)


    async def _golden_setup_screener_loop(self):
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
                watchlist_list = list(watchlist_set)
                random.shuffle(watchlist_list)
                logger.debug(f"Начинаю сканирование {len(watchlist_list)} монет в случайном порядке...")
                tasks = [strategies.golden_strategy(self, symbol) for symbol in watchlist_list]
                await asyncio.gather(*tasks)
            except Exception as e:
                logger.error(f"Ошибка в цикле сканера Golden Setup: {e}", exc_info=True)

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
        if is_quiet_market_window():
            if 'golden' in source.lower():
                return True, "Межсессионка: квоты для GOLDEN отключены."

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


    # --- ИСПРАВЛЕННАЯ ВЕРСИЯ ---
    # Этот фильтр теперь учитывает тип стратегии. Контртрендовые фильтры
    # не будут применяться к трендовым и пробойным сигналам.
    async def _entry_guard(self, symbol: str, side: str, features: dict, candidate: dict) -> tuple[bool, str]:
        cfg = self.user_data.get("entry_guard_settings", config.ENTRY_GUARD)
        now = time.time()
        
        # 1. Общие проверки для всех стратегий
        self.pending_policy_vectors.pop(symbol, None)
        if spread := float(features.get("spread_pct", 0.0)) > cfg.get("MAX_SPREAD_PCT", 0.25):
            return False, f"spread {spread:.2f}% > {cfg['MAX_SPREAD_PCT']:.2f}%"

        # 2. Контекстно-зависимые проверки
        source = candidate.get("source", "").lower()
        is_counter_trend_strategy = any(k in source for k in ['squeeze', 'liquidation', 'fade'])

        # Применяем фильтр "анти-погони" только для контртрендовых стратегий
        if is_counter_trend_strategy:
            cd_key = (symbol, side)
            if now < self.momentum_cooldown_until.get(cd_key, 0.0):
                left = int(self.momentum_cooldown_until[cd_key] - now)
                return False, f"cooldown {left}s"

            pct1m = float(features.get("pct1m", 0.0))
            pct5m = float(features.get("pct5m", 0.0))
            pump1 = cfg.get("PUMP_BLOCK_1M_PCT", 1.2)
            pump5 = cfg.get("PUMP_BLOCK_5M_PCT", 3.0)
            dump1 = cfg.get("DUMP_BLOCK_1M_PCT", 1.2)
            dump5 = cfg.get("DUMP_BLOCK_5M_PCT", 3.0)
            
            CVD1m = float(features.get("CVD1m", 0.0))
            CVD5m = float(features.get("CVD5m", 0.0))
            dOI1m = float(features.get("dOI1m", 0.0))
            dOI5m = float(features.get("dOI5m", 0.0))

            def aligned_up():
                return (CVD1m > 0 or CVD5m > 0) and (dOI1m > 0 or dOI5m > 0)

            def aligned_down():
                return (CVD1m < 0 or CVD5m < 0) and (dOI1m < 0 or dOI5m < 0)

            if side == "Sell":
                if (pct1m > pump1 or pct5m > pump5) and aligned_up():
                    self.momentum_cooldown_until[cd_key] = now + cfg.get("MOMENTUM_COOLDOWN_SEC", 90)
                    return False, "anti-chase-pump"
            else: # side == "Buy"
                if (pct1m < -dump1 or pct5m < -dump5) and aligned_down():
                    self.momentum_cooldown_until[cd_key] = now + cfg.get("MOMENTUM_COOLDOWN_SEC", 90)
                    return False, "anti-chase-dump"
            
        ok_policy, policy_reason = self._ml_policy_entry_check(symbol, side, features, candidate)
        if not ok_policy:
            return False, policy_reason
        return True, policy_reason

    def _build_policy_vector(self, features: dict, side: str) -> Optional[np.ndarray]:
        policy = getattr(self, "ml_policy", None)
        if not policy or not getattr(policy, "enabled", False):
            return None
        if not features:
            return None
        try:
            vector = [utils.safe_to_float(features.get(key, 0.0)) for key in config.FEATURE_KEYS]
        except Exception as exc:
            logger.debug(f"[PolicyVector] Ошибка извлечения признаков: {exc}")
            return None
        vector.append(1.0 if str(side).lower() == "buy" else -1.0)
        arr = np.asarray(vector, dtype=np.float32)
        if arr.size != self.policy_feature_dim:
            logger.debug(f"[PolicyVector] Размерность {arr.size} != ожидаемой {self.policy_feature_dim}.")
            return None
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _ml_policy_entry_check(self, symbol: str, side: str, features: dict, candidate: dict) -> tuple[bool, str]:
        policy = getattr(self, "ml_policy", None)
        if not policy or not getattr(policy, "enabled", False):
            return True, "ok"
        source = candidate.get("source", "")
        if not policy.should_apply(source):
            return True, "ok"
        vector = self._build_policy_vector(features, side)
        if vector is None:
            return True, "policy vector unavailable"
        score = policy.score(vector)
        meta = {
            "vector": vector,
            "score": score,
            "side": side,
            "source": source,
            "ts": time.time(),
        }
        self.pending_policy_vectors[symbol] = meta
        if score is not None:
            features["ml_policy_score"] = score
            candidate["ml_policy_score"] = score
            threshold = policy.get_entry_threshold()
            if score < threshold:
                return False, f"ml-policy score {score:.2f}<{threshold:.2f}"
            return True, f"ml-policy score {score:.2f}"
        if policy.bypass_when_unfit:
            return True, "ml policy not ready (bypass)"
        return False, "ml policy not ready"

    def _register_policy_outcome(self, symbol: str, pnl_pct: float) -> None:
        policy = getattr(self, "ml_policy", None)
        if not policy or not getattr(policy, "enabled", False):
            self.learning_positions.pop(symbol, None)
            self.pending_policy_records.pop(symbol, None)
            return
        record = self.learning_positions.pop(symbol, None)
        if record is None:
            record = self.pending_policy_records.pop(symbol, None)
        if not record:
            return
        if not policy.should_apply(record.get("source", "")):
            return
        vector = record.get("vector")
        if vector is None:
            return
        try:
            policy.update(vector, pnl_pct)
        except Exception as exc:
            logger.warning(f"[PolicyLearner] Ошибка при обновлении исхода {symbol}: {exc}")

    def _policy_exit_decision(self, symbol: str, pos: dict, features: dict) -> bool:
        policy = getattr(self, "ml_policy", None)
        if not policy or not getattr(policy, "enabled", False):
            return False
        source = pos.get("source", "")
        if not policy.should_apply(source):
            return False
        side = pos.get("side")
        vector = self._build_policy_vector(features or {}, side)
        if vector is None:
            return False
        score = policy.score(vector)
        if score is None:
            return False
        pos["ml_policy_score"] = score
        record = self.learning_positions.get(symbol)
        if record is not None:
            record["latest_score"] = score
        threshold = policy.get_exit_threshold()
        if score >= threshold:
            return False
        min_hold = policy.get_min_hold_sec()
        open_ts = utils.safe_to_float(pos.get("open_ts", 0.0))
        if min_hold > 0 and open_ts > 0 and (time.time() - open_ts) < min_hold:
            return False
        logger.info(f"[ML-Policy] {symbol}: score {score:.2f} ниже exit-порога {threshold:.2f}.")
        return True


    def _load_wall_memory(self):
        try:
            if self._wall_memory_file.exists() and self._wall_memory_file.stat().st_size > 0:
                with open(self._wall_memory_file, 'rb') as f:
                    self.dom_wall_memory = pickle.load(f)
                logger.info(f"🧠 [Память Стен] Успешно загружено {sum(len(v) for v in self.dom_wall_memory.values())} уровней из {self._wall_memory_file}")
        except FileNotFoundError:
            logger.info(f"🧠 [Память Стен] Файл {self._wall_memory_file} не найден. Начинаем с чистой памяти.")
        except Exception as e:
            logger.error(f"🧠 [Память Стен] КРИТИЧЕСКАЯ ОШИБКА загрузки: {e}", exc_info=True)

    # def _save_wall_memory(self):
    #     try:
    #         with open(self._wall_memory_file, 'wb') as f:
    #             pickle.dump(self.dom_wall_memory, f)
    #         logger.debug(f"🧠 [Память Стен] Данные сохранены в {self._wall_memory_file}")
    #     except Exception as e:
    #         logger.warning(f"🧠 [Память Стен] Ошибка сохранения: {e}", exc_info=True)


    async def _save_wall_memory_locked(self):
        """Запись памяти стен на диск. Предполагает, что lock уже захвачен."""
        try:
            memory_copy = self.dom_wall_memory.copy()

            def dump_pickle():
                with open(self._wall_memory_file, 'wb') as f:
                    pickle.dump(memory_copy, f)

            await asyncio.to_thread(dump_pickle)
            logger.debug(f"🧠 [Память Стен] Данные сохранены в {self._wall_memory_file}")
        except RuntimeError as e:
            if "changed size during iteration" in str(e):
                logger.warning(f"🧠 [Память Стен] Ошибка сохранения (гонка потоков): {e}. Повторная попытка...")
                await asyncio.sleep(0.1)
                await self._save_wall_memory_locked()
            else:
                logger.warning(f"🧠 [Память Стен] Ошибка сохранения: {e}", exc_info=True)
        except Exception as e:
            logger.warning(f"🧠 [Память Стен] Ошибка сохранения: {e}", exc_info=True)

    async def _save_wall_memory(self):
        """Асинхронное и потокобезопасное сохранение памяти стен."""
        async with self.wall_memory_lock:
            await self._save_wall_memory_locked()


    # async def _wall_memory_save_loop(self, interval: int = 300):
    #     while True:
    #         await asyncio.sleep(interval)
    #         try:
    #             await asyncio.to_thread(self._save_wall_memory)
    #         except Exception as e:
    #             logger.error(f"🧠 [Память Стен] КРИТИЧЕСКАЯ ОШИБКА в цикле сохранения: {e}", exc_info=True)


    async def _wall_memory_save_loop(self, interval: int = 300):
        while True:
            await asyncio.sleep(interval)
            try:
                # --- ИЗМЕНЕНИЕ: Убираем to_thread, вызываем async-функцию напрямую ---
                await self._save_wall_memory()
            except Exception as e:
                logger.error(f"🧠 [Память Стен] КРИТИЧЕСКАЯ ОШИБКА в цикле сохранения: {e}", exc_info=True)



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
        failures = 0
        while True:
            delay = interval if failures == 0 else min(interval + failures * 5, interval * 4)
            await asyncio.sleep(delay)
            try:
                await self.update_open_positions()
                failures = 0
            except Exception as e:
                failures += 1
                logger.error(f"Ошибка в цикле синхронизации позиций: {e}", exc_info=True)
                backoff = min(120.0, 5.0 * failures)
                await asyncio.sleep(backoff)

    async def wallet_loop(self):
        wallet_logger = logging.getLogger("wallet_state")
        failures = 0
        while True:
            try:
                resp = await asyncio.to_thread(
                    lambda: self.session.get_wallet_balance(accountType="UNIFIED")
                )
                wallet_list = resp.get("result", {}).get("list", [])
                if wallet_list:
                    account = wallet_list[0]
                    total_im = utils.safe_to_float(account.get("totalInitialMargin"))
                    if abs(total_im - self.current_total_volume) > 1e-6:
                        self.current_total_volume = total_im
                        wallet_logger.info(f"[User {self.user_id}] IM={total_im:.2f}")

                    snapshot = {
                        "totalEquity": utils.safe_to_float(account.get("totalEquity") or account.get("equity")),
                        "availableBalance": utils.safe_to_float(account.get("availableBalance")),
                        "usedMargin": utils.safe_to_float(account.get("usedMargin") or total_im),
                    }
                    coins = account.get("coin")
                    if isinstance(coins, list):
                        for coin in coins:
                            if coin.get("coin") in ("USDT", "USDC"):
                                snapshot.setdefault("totalEquity", utils.safe_to_float(coin.get("equity") or coin.get("walletBalance")))
                                snapshot.setdefault("availableBalance", utils.safe_to_float(coin.get("availableBalance") or coin.get("availableToBorrow")))
                                snapshot.setdefault("usedMargin", utils.safe_to_float(coin.get("usedToLoan")))
                                break
                    self._snapshot_wallet_state({k: v for k, v in snapshot.items() if v is not None})
                failures = 0
                sleep_s = 10.0 if self.current_total_volume > 0 else 30.0
            except Exception as e:
                failures += 1
                wallet_logger.debug(f"[wallet_loop] error: {e}")
                sleep_s = min(60.0, 10.0 + failures * 5.0)
            await asyncio.sleep(sleep_s)

    async def execute_trade_entry(self, candidate: dict, features: dict):
        symbol = candidate.get("symbol")
        side = candidate.get("side")  # "Buy" / "Sell"
        source = candidate.get("source", "N/A")
        source_comment = candidate.get("justification", source)

        logger.info(f"[EXEC_FLOW] Begin entry pipeline: {symbol}/{side} src={source}")

        # 1) Фичи/гвард на вход
        if not features:
            features = await self.extract_realtime_features(symbol)
        if features is None:
            features = {}

        # --- ML: прокинем уверенность в features и сохраним в состоянии бота позже ---
        ml_conf = 0.5
        try:
            ml_conf = float(
                (features or {}).get("ml_confidence")
                or candidate.get("ml_conf")
                or 0.5
            )
        except Exception:
            ml_conf = 0.5
        features["ml_confidence"] = ml_conf
        # --------------------------------------------------------------------------- 

        # --- проверка re-entry cooldown ---
        cooldown = max(0.0, float(getattr(self, "reentry_cooldown_sec", 0.0)))
        last_close_ts = self._last_close_ts.get(symbol, 0.0)
        now_ts = time.time()
        if cooldown > 0 and last_close_ts and (now_ts - last_close_ts) < cooldown:
            remaining = cooldown - (now_ts - last_close_ts)
            logger.info(
                f"[ENTRY_GUARD] {symbol}/{side} отклонён: cooldown {remaining:.1f}s после закрытия."
            )
            return

        # --- изменено: передаём candidate внутрь _entry_guard для большего контекста ---
        ok, reason = await self._entry_guard(symbol, side, features, candidate)
        if not ok:
            logger.warning(f"[ENTRY_GUARD] {symbol}/{side} отклонён: {reason}")
            self.pending_policy_vectors.pop(symbol, None)
            # на этом этапе мы ещё ничего не резервировали
            return

        logger.info(f"[EXEC_FLOW] Guard passed {symbol}/{side}, reason={reason}")
        policy_meta = self.pending_policy_vectors.pop(symbol, None)
        policy_score = None
        if policy_meta:
            policy_score = policy_meta.get("score")
            policy_meta["features_snapshot_ts"] = time.time()
            self.pending_policy_records[symbol] = dict(policy_meta)
            if policy_score is not None:
                features["ml_policy_score"] = policy_score
                candidate["ml_policy_score"] = policy_score

        # 2) Целевой объём и лимиты биржи
        base_volume = float(self.POSITION_VOLUME)
        target_volume = self._determine_position_volume(symbol, base_volume, features)
        logger.info(f"[EXEC_FLOW] Volume target {symbol}: base≈{target_volume:.2f} USD (pre-cap)")

        await self.ensure_symbol_meta(symbol)

        # 3) Цена и метаданные инструмента
        last_price = await self._resolve_last_price(symbol, features)
        if last_price <= 0:
            logger.warning(f"[EXECUTE_REJECT] {symbol}: нет котировки (features/ticker/orderbook).")
            self.pending_policy_records.pop(symbol, None)
            return

        step = float(self.qty_step_map.get(symbol, 0.001) or 0.001)
        min_qty = float(self.min_qty_map.get(symbol, step) or step)
        max_qty = utils.safe_to_float(self.max_qty_map.get(symbol))

        # 4) Расчёт и округление qty
        raw_qty = target_volume / last_price if last_price > 0 else 0.0
        qty_dec = self._round_to_step(symbol, raw_qty)
        min_qty_dec = self._round_to_step(symbol, max(min_qty, step), mode="ceil")
        if qty_dec < min_qty_dec:
            qty_dec = min_qty_dec

        if max_qty > 0:
            max_qty_dec = self._round_to_step(symbol, max_qty, mode="floor")
            if max_qty_dec <= 0:
                logger.warning(f"[EXECUTE_REJECT] {symbol}: биржевой максимум объёма {max_qty} несовместим с шагом {step}.")
                self.pending_policy_records.pop(symbol, None)
                return
            if qty_dec > max_qty_dec:
                logger.warning(f"[EXEC_FLOW] {symbol}: qty {float(qty_dec):.6f} превышает биржевой максимум {max_qty}. Применяю кап.")
                qty_dec = max_qty_dec

        qty = float(qty_dec)

        if qty <= 0:
            logger.warning(f"[EXECUTE_REJECT] {symbol}: qty<=0 после округления (last={last_price}, target={target_volume}).")
            self.pending_policy_records.pop(symbol, None)
            return

        volume_to_open = qty * last_price
        effective_total_vol = await self.get_effective_total_volume()
        if effective_total_vol + volume_to_open > self.MAX_TOTAL_VOLUME:
            logger.warning(f"[EXECUTE_REJECT] {symbol}: лимит портфеля. Попытка={volume_to_open:.2f}, Лимит={self.MAX_TOTAL_VOLUME:.2f}")
            self.pending_policy_records.pop(symbol, None)
            return

        logger.info(f"[EXEC_FLOW] Volume resolved {symbol}: final≈{volume_to_open:.2f} USD, effective_total={effective_total_vol:.2f}")
        logger.info(
            f"[EXEC_FLOW] Qty resolved {symbol}: raw={raw_qty:.6f}, rounded={qty}, min={min_qty_dec}, "
            f"last_price={last_price:.6f}, max={max_qty if max_qty > 0 else 'inf'}"
        )

        trailing_mode = str(
            candidate.get("trailing_mode")
            or self.user_data.get("trailing_mode")
            or config.ACTIVE_TRAILING_MODE
            or "simple_gap"
        )

        # 5) Метаданные позиции (дополняем ML-уверенностью)
        meta_payload = {
            "wall_rating": candidate.get("wall_rating"),
            "wall_hold_ratio": candidate.get("wall_hold_ratio"),
            "wall_zscore": candidate.get("wall_zscore"),
            "wall_size": candidate.get("wall_size"),
            "confidence": candidate.get("confidence"),
            "trailing_mode": trailing_mode,
            "ml_confidence": ml_conf,  # <--- добавлено
            "ml_policy_score": policy_score,
        }
        meta_payload = {k: v for k, v in meta_payload.items() if v is not None}
        if meta_payload:
            self.pending_position_meta[symbol] = meta_payload
        else:
            self.pending_position_meta.pop(symbol, None)

        # 6) Атомарное резервирование pending-объёма + повторная проверка лимита ПОРТФЕЛЯ внутри лока
        async with self.pending_orders_lock:
            # дубликаты/состязание
            if symbol in self.open_positions or symbol in self.pending_orders:
                logger.warning(f"[EXEC_FLOW] {symbol}: double-entry guard triggered (open_positions={symbol in self.open_positions}, pending={symbol in self.pending_orders}).")
                self.pending_position_meta.pop(symbol, None)
                self.pending_policy_records.pop(symbol, None)
                return

            # повторная проверка объёма портфеля (чтобы убрать гонку)
            open_notional_locked = self._calc_open_positions_notional()
            pending_notional_locked = sum(abs(v) for v in self.pending_orders.values())
            effective_total_vol_locked = open_notional_locked + pending_notional_locked
            if effective_total_vol_locked + volume_to_open > self.MAX_TOTAL_VOLUME:
                logger.warning(f"[EXEC_FLOW] {symbol}: portfolio limit inside lock. current={effective_total_vol_locked:.2f}, attempt={volume_to_open:.2f}, limit={self.MAX_TOTAL_VOLUME:.2f}")
                self.pending_position_meta.pop(symbol, None)
                self.pending_policy_records.pop(symbol, None)
                return

            # резервируем pending
            self.pending_orders[symbol] = volume_to_open
            self.pending_timestamps[symbol] = time.time()
            logger.info(f"[EXEC_FLOW] {symbol}: pending reserved volume={volume_to_open:.2f}")

            # --- ML: сохраняем уверенность на уровне позиции (для трейлинга и пр.) ---
            if not hasattr(self, "position_ml_confidence"):
                self.position_ml_confidence = {}
            self.position_ml_confidence[symbol] = float(ml_conf)
            # -------------------------------------------------------------------------

        # 7) Сохраняем опорную цену
        self.pending_open_exec[symbol] = {"side": side, "price": last_price, "ts": time.time()}
        logger.info(f"[EXEC_FLOW] {symbol}: pending_open_exec stored with price={last_price:.6f}")

        qty_str = format(qty_dec.normalize(), "f")

        try:
            qty_notional = qty * last_price
            logger.info(f"🚀 [EXECUTION] {symbol} {side} — qty={qty_str} (~{qty_notional:.2f} USDT). src={source}")
            await self.place_unified_order(
                symbol=symbol,
                side=side,
                qty=qty_dec,
                order_type="Market",
                comment=source_comment
            )
            self.pending_strategy_comments[symbol] = source
            if 'stop_loss_price_base' in candidate:
                self.pending_strategy_comments[f"{symbol}_sl_base"] = candidate['stop_loss_price_base']
            self.last_entry_ts[symbol] = time.time()
            self._schedule_fail_fast_guard(symbol, side.lower(), source)

        except Exception as e:
            logger.error(f"[EXECUTE_CRITICAL] {symbol}: ошибка при входе — {e}", exc_info=True)
            # Чистим все следы pending и ML при неудаче
            async with self.pending_orders_lock:
                self.pending_orders.pop(symbol, None)
                self.pending_timestamps.pop(symbol, None)
            self.pending_open_exec.pop(symbol, None)
            self.pending_position_meta.pop(symbol, None)
            if hasattr(self, "position_ml_confidence"):
                self.position_ml_confidence.pop(symbol, None)
            self.pending_policy_records.pop(symbol, None)


    async def execute_trade_entry_golden_setup(
        self,
        symbol: str,
        side: str,
        *,
        reason: str = "GOLDEN DIRECT",
        mkt_features: dict | None = None,
    ):
        """
        Прямой вход по золотому сетапу: сразу MARKET-ордер.
        Без следопыта и прочих гейтов. Детальные логи.
        """
        lg = getattr(self, "logger", logger)  # <- безопасный логгер

        def _f(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return default

        try:
            # --- 0) Объём сделки и лимиты портфеля ---
            per_trade_usdt = _f(getattr(self, "POSITION_VOLUME", None) or getattr(self, "position_volume", None) or 0.0) or 100.0
            max_total = _f(getattr(self, "MAX_TOTAL_VOLUME", None) or getattr(self, "max_total_volume", None) or 0.0) or 1e12

            portfolio_now = 0.0
            # пробуем разные методы, что есть в твоём боте
            for name in ("get_effective_total_volume", "get_total_positions_notional", "get_portfolio_notional", "_get_total_positions_notional"):
                fn = getattr(self, name, None)
                if fn:
                    try:
                        portfolio_now = await fn() if asyncio.iscoroutinefunction(fn) else fn()
                        portfolio_now = _f(portfolio_now, 0.0)
                        break
                    except Exception as e:
                        lg.debug(f"[GS_WARN] portfolio calc via {name} failed: {e}")

            will_open = max(0.0, min(per_trade_usdt, max_total - portfolio_now))
            lg.info(
                f"[GS_EXEC] {symbol}/{side}: per_trade={per_trade_usdt:.2f}, "
                f"portfolio_now={portfolio_now:.2f}, will_open={will_open:.2f}, limit={max_total:.2f}"
            )
            if will_open <= 0:
                lg.info(f"[GS_REJECT] {symbol}/{side}: не осталось лимита портфеля.")
                return None

            # --- 1) Цена с fallback'ами ---
            last_price = None
            for name in ("_resolve_last_price", "get_last_price", "_get_last_price"):
                fn = getattr(self, name, None)
                if fn:
                    try:
                        lp = await fn(symbol) if asyncio.iscoroutinefunction(fn) else fn(symbol)
                        last_price = _f(lp, 0.0)
                        if last_price > 0:
                            break
                    except Exception as e:
                        lg.debug(f"[GS_WARN] last_price via {name} failed: {e}")

            if not last_price or last_price <= 0:
                # тикер / свечи из WS
                lp = 0.0
                sws = getattr(self, "shared_ws", None)
                if sws and getattr(sws, "ticker_data", None):
                    td = sws.ticker_data.get(symbol) or {}
                    lp = _f(td.get("lastPrice") or td.get("lp") or 0.0, 0.0)
                if (not lp or lp <= 0) and sws and getattr(sws, "candles_data", None):
                    arr = (sws.candles_data.get(symbol) or [])
                    if arr:
                        c = arr[-1]
                        lp = _f(c.get("closePrice") or c.get("close") or c.get("c") or 0.0, 0.0)
                last_price = lp

            if not last_price or last_price <= 0:
                lg.warning(f"[GS_REJECT] {symbol}/{side}: не получили корректную цену (last_price).")
                return None

            # --- 2) Метаданные символа / шаги ---
            if hasattr(self, "ensure_symbol_meta"):
                try:
                    await self.ensure_symbol_meta(symbol)
                except Exception as e:
                    lg.debug(f"[GS_WARN] ensure_symbol_meta failed: {e}")

            qty_step_map = getattr(self, "qty_step_map", {}) or {}
            min_qty_map = getattr(self, "min_qty_map", {}) or {}
            symbol_meta = getattr(self, "symbol_meta", {}) or {}

            step = _f(qty_step_map.get(symbol) or 0.0, 0.0)
            min_qty = _f(min_qty_map.get(symbol) or 0.0, 0.0)
            category = (symbol_meta.get(symbol) or {}).get("category") or "linear"

            # --- 3) Кол-во и округление ---
            raw_qty = will_open / last_price
            qty = raw_qty
            if step > 0:
                qty = math.floor(qty / step) * step

            if qty <= 0:
                lg.info(f"[GS_REJECT] {symbol}/{side}: qty<=0 после округления (raw={raw_qty:.8f}, step={step}).")
                return None

            if min_qty > 0 and qty < min_qty:
                lg.info(f"[GS_REJECT] {symbol}/{side}: qty {qty:.8f} < min_qty {min_qty:.8f}.")
                return None

            notional = qty * last_price
            lg.info(
                f"[GS_ORDER] {symbol}/{side}: MARKET | price≈{last_price:.6g} | qty={qty:.8f} "
                f"| notional≈{notional:.2f} USDT | category={category} | {reason}"
            )

            # --- 4) Отправка MARKET-ордера ---
            resp = await self.place_unified_order(
                symbol=symbol,
                side=side,
                order_type="Market",
                qty=qty,
                category=category,
                reduce_only=False,
            )

            ok = isinstance(resp, dict) and str(resp.get("retCode", "")) in ("0", "OK", "Success")
            if ok:
                rid = (resp.get("result") or {}).get("orderLinkId") or (resp.get("result") or {}).get("orderId")
                lg.info(f"✅ [GS_DONE] {symbol}/{side}: ордер размещен | qty={qty:.8f} | id={rid}")
            else:
                lg.error(f"❌ [GS_FAIL] {symbol}/{side}: ответ биржи: {resp}")

            # лёгкий cooldown на символ, чтобы не спамить
            if hasattr(self, "_last_golden_ts"):
                self._last_golden_ts[symbol] = time.time()

            return resp

        except Exception as e:
            lg.error(f"❌ [GS_ERROR] {symbol}/{side}: {e}", exc_info=True)
            return None





    async def _pre_flight_revalidate_breakout(self, candidate: dict) -> bool:
        """
        Проверяем, что цена действительно принята за стеной:
        - Для Sell: last <= wall - 1 тик (и не дрейфует назад > MAX_DRIFT_TICKS).
        - Для Buy:  last >= wall + 1 тик (и не дрейфует назад > MAX_DRIFT_TICKS).
        Ждём до RETEST_SEC, опрашивая раз в ~250мс.
        """
        cfg = self.user_data.get("dom_squeeze_settings", config.DOM_SQUEEZE_STRATEGY)
        pre = cfg.get("BREAKOUT_PREFLIGHT", {})
        RETEST_SEC       = float(pre.get("RETEST_SEC", 3.0))
        MAX_DRIFT_TICKS  = int(pre.get("MAX_DRIFT_TICKS", 2))

        symbol = candidate["symbol"]
        side   = str(candidate["side"]).lower()
        wall   = utils.safe_to_float(candidate.get("wall_price", 0.0))
        tick   = float(self.price_tick_map.get(symbol, 1e-6) or 1e-6)

        deadline = time.time() + RETEST_SEC
        accepted = False

        while time.time() < deadline:
            last = self.safe_last_price(symbol)
            if last <= 0:
                await asyncio.sleep(0.25)
                continue

            if side == "sell":
                # нужно быть ниже стены хотя бы на тик; если задернулись выше стены на n тиков — отменяем
                if last <= wall - tick:
                    accepted = True
                    break
                if last >= wall + MAX_DRIFT_TICKS * tick:
                    return False
            else:
                if last >= wall + tick:
                    accepted = True
                    break
                if last <= wall - MAX_DRIFT_TICKS * tick:
                    return False

            await asyncio.sleep(0.25)

        return accepted


    async def _post_entry_failsafe_breakout(self, candidate: dict) -> None:
        """
        Первые WINDOW_SEC секунд после входа: если цена делает RECLAIM стены против нас
        (возвращается за стену на RECLAIM_TICKS), срочно выходим через аварийный SL.
        """
        cfg = self.user_data.get("dom_squeeze_settings", config.DOM_SQUEEZE_STRATEGY)
        fs = cfg.get("BREAKOUT_FAILSAFE", {})
        WINDOW_SEC    = float(fs.get("WINDOW_SEC", 25.0))
        RECLAIM_TICKS = int(fs.get("RECLAIM_TICKS", 3))

        symbol = candidate["symbol"]
        side   = str(candidate["side"]).lower()
        wall   = utils.safe_to_float(candidate.get("wall_price", 0.0))
        tick   = float(self.price_tick_map.get(symbol, 1e-6) or 1e-6)

        t_end = time.time() + WINDOW_SEC

        while time.time() < t_end:
            pos = self.open_positions.get(symbol) or {}
            if not pos or str(pos.get("side", "")).lower() != side:
                return  # позиция уже закрыта/перевернута

            last = self.safe_last_price(symbol)
            if last <= 0:
                await asyncio.sleep(0.3)
                continue

            if side == "sell":
                # reclaim выше стены → выходим
                if last >= wall + RECLAIM_TICKS * tick:
                    failsafe = max(last + tick, wall + (RECLAIM_TICKS - 1) * tick)
                    fs_rounded = self._round_to_tick(symbol, failsafe, "sell")
                    logger.warning(f"[{symbol}] ❌ Breakout fail-safe: reclaim выше стены. SL→{fs_rounded:.6f}")
                    await self.set_or_amend_stop_loss(fs_rounded, symbol=symbol)
                    return
            else:
                if last <= wall - RECLAIM_TICKS * tick:
                    failsafe = min(last - tick, wall - (RECLAIM_TICKS - 1) * tick)
                    fs_rounded = self._round_to_tick(symbol, failsafe, "buy")
                    logger.warning(f"[{symbol}] ❌ Breakout fail-safe: reclaim ниже стены. SL→{fs_rounded:.6f}")
                    await self.set_or_amend_stop_loss(fs_rounded, symbol=symbol)
                    return

    def _schedule_fail_fast_guard(self, symbol: str, side: str, source: str) -> None:
        cfg = self._get_fail_fast_config()
        if not cfg.get("ENABLED", True):
            return
        if "golden_setup" not in (source or "").lower():
            return
        task = self._fail_fast_tasks.get(symbol)
        if task and not task.done():
            task.cancel()
        self._fail_fast_tasks[symbol] = asyncio.create_task(
            self._fail_fast_guard(symbol, side.lower())
        )

    def _get_fail_fast_config(self) -> dict:
        base = copy.deepcopy(getattr(config, "GOLDEN_FAIL_FAST", {}))
        user_cfg = {}
        try:
            user_cfg = (self.user_data or {}).get("GOLDEN_FAIL_FAST") or {}
        except Exception:
            user_cfg = {}
        if user_cfg:
            base.update(user_cfg)
        return base

    async def _fail_fast_guard(self, symbol: str, side: str) -> None:
        cfg = self._get_fail_fast_config()
        if not cfg.get("ENABLED", True):
            return
        try:
            wait_for_fill = float(cfg.get("WAIT_FOR_FILL_SEC", 5.0))
            deadline = time.time() + wait_for_fill
            while time.time() < deadline:
                pos = self.open_positions.get(symbol)
                if pos and str(pos.get("side", "")).lower() == side:
                    break
                await asyncio.sleep(0.2)
            else:
                return

            window_sec = float(cfg.get("WINDOW_SEC", 30.0))
            loss_pct = float(cfg.get("LOSS_PCT", 0.6))
            min_positive_pct = float(cfg.get("MIN_POSITIVE_PCT", 0.2))
            breakeven_trigger = float(cfg.get("BREAKEVEN_TRIGGER_PCT", 0.0))
            breakeven_buffer_pct = float(cfg.get("BREAKEVEN_BUFFER_PCT", 0.0))
            max_hold_sec = float(cfg.get("MAX_HOLD_SEC", 0.0))
            base_leverage = utils.safe_to_float(
                getattr(self, "symbol_leverage_map", {}).get(symbol)
                or self.user_data.get("leverage")
                or getattr(self, "leverage", 1.0)
            )
            base_leverage = max(1.0, base_leverage)
            start_ts = time.time()

            while True:
                pos = self.open_positions.get(symbol)
                if not pos or str(pos.get("side", "")).lower() != side:
                    return

                lev = utils.safe_to_float(pos.get("leverage")) if pos else 0.0
                if not lev:
                    lev = base_leverage
                lev = max(1.0, lev)

                avg_price = self._resolve_avg_price(symbol, pos)
                last_price = self.safe_last_price(symbol)
                if avg_price <= 0 or last_price <= 0:
                    await asyncio.sleep(0.5)
                    continue

                roi_price_pct = ((last_price - avg_price) / avg_price * 100.0) if side == "buy" else ((avg_price - last_price) / avg_price * 100.0)
                roi_pct = roi_price_pct * lev
                elapsed = time.time() - start_ts

                if max_hold_sec > 0 and elapsed >= max_hold_sec:
                    logger.warning(f"[{symbol}] Fail-fast: max hold {max_hold_sec}s достигнут. Принудительное закрытие позиции.")
                    await self.close_position(symbol, reason="Golden fail-fast max hold")
                    return

                if elapsed >= window_sec:
                    if roi_pct <= -loss_pct:
                        logger.warning(f"[{symbol}] Fail-fast: ROI={roi_pct:.2f}% ≤ -{loss_pct}%. Принудительно закрываем позицию.")
                        await self.close_position(symbol, reason="Golden fail-fast loss")
                        return

                    if breakeven_trigger > 0:
                        if roi_pct >= breakeven_trigger:
                            already_armed = bool(pos.get("fail_fast_be_armed"))
                            stop_price = 0.0
                            if side == "buy":
                                stop_price = avg_price * (1.0 + breakeven_buffer_pct / 100.0)
                            else:
                                stop_price = avg_price * (1.0 - breakeven_buffer_pct / 100.0)

                            stop_price = self._round_to_tick(symbol, stop_price, side)

                            if stop_price > 0 and (not already_armed or not math.isclose(stop_price, pos.get("fail_fast_be_price", 0.0), rel_tol=0.0, abs_tol=(self._get_tick(symbol) or 1e-6))):
                                await self.set_or_amend_stop_loss(stop_price, symbol=symbol)
                                pos["fail_fast_be_armed"] = True
                                pos["fail_fast_be_price"] = stop_price
                                logger.info(
                                    f"[{symbol}] Fail-fast: ROI={roi_pct:.2f}% (price move {roi_price_pct:.2f}% при плече x{lev:.1f}) ≥ {breakeven_trigger:.2f}%. "
                                    f"Переносим стоп в безубыточность ({stop_price:.6f}, буфер {breakeven_buffer_pct:.3f}%)."
                                )
                        # Если порог не достигнут — просто наблюдаем дальше без подтяжки к входу.
                    else:
                        if roi_pct < min_positive_pct:
                            tick = self._get_tick(symbol) or 1e-6
                            if side == "buy":
                                target = self._round_to_tick(symbol, avg_price - tick, "buy")
                            else:
                                target = self._round_to_tick(symbol, avg_price + tick, "sell")
                            logger.info(f"[{symbol}] Fail-fast: ROI={roi_pct:.2f}% < {min_positive_pct}%. Стоп подтянут к входу ({target:.6f}).")
                            await self.set_or_amend_stop_loss(target, symbol=symbol)
                            return

                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            current_task = asyncio.current_task()
            task = self._fail_fast_tasks.get(symbol)
            if task is current_task:
                self._fail_fast_tasks.pop(symbol, None)

            await asyncio.sleep(0.4)

    async def _enter_breakout_on_retest(self, candidate: dict, current_features: dict | None = None) -> None:
        """
        Предполагается, что _pre_flight_revalidate_breakout() уже пройден.
        Ждём откат к диапазону ретеста у стены и формальный отбой на BOUNCE_TICKS,
        затем отправляем вход (execute_trade_entry). Если ретеста нет — не входим.
        """
        cfg = self.user_data.get("dom_squeeze_settings", config.DOM_SQUEEZE_STRATEGY)
        rcfg = cfg.get("RETEST_SETTINGS", {})

        symbol           = candidate["symbol"]
        side             = str(candidate["side"]).lower()
        wall             = utils.safe_to_float(candidate.get("wall_price", 0.0))
        tick             = float(self.price_tick_map.get(symbol, 1e-6) or 1e-6)

        WAIT_WINDOW_SEC  = float(rcfg.get("WAIT_WINDOW_SEC", 20.0))   # сколько ждём ретест
        BAND_TICKS       = int(rcfg.get("BAND_TICKS", 2))             # ширина «зоны ретеста» вокруг стены
        BOUNCE_TICKS     = int(rcfg.get("BOUNCE_TICKS", 2))           # отбой на столько тиков в сторону входа
        BOUNCE_CONFIRM_S = float(rcfg.get("BOUNCE_CONFIRM_SEC", 1.0)) # сколько секунд держится отбой
        MAX_SPREAD_TICKS = int(rcfg.get("MAX_SPREAD_TICKS", 5))       # не входим, если спрэд разъехался

        t_deadline   = time.time() + WAIT_WINDOW_SEC
        in_band_since = 0.0

        def in_retest_band(p: float) -> bool:
            lo = wall - BAND_TICKS * tick
            hi = wall + BAND_TICKS * tick
            return lo <= p <= hi

        while time.time() < t_deadline:
            # если позиция уже открыта (другой поток), выходим
            pos = self.open_positions.get(symbol)
            if pos and str(pos.get("side", "")).lower() == side:
                return

            last = self.safe_last_price(symbol)
            if last <= 0:
                await asyncio.sleep(0.2)
                continue

            # простая проверка на «рваный» спред — не лезем
            tick = float(self.price_tick_map.get(symbol) or 0.0) or 1e-6
            # last_price уже есть в контексте
            last = self.safe_last_price(symbol)
            best_ask = utils.safe_to_float(self.best_ask_map.get(symbol, 0.0)) or last
            best_bid = utils.safe_to_float(self.best_bid_map.get(symbol, 0.0)) or last
            mid = (best_ask + best_bid) / 2 if (best_ask > 0 and best_bid > 0) else last

            if best_ask and best_bid:
                spread_ticks = int(round((best_ask - best_bid) / tick))
                if spread_ticks > MAX_SPREAD_TICKS:
                    await asyncio.sleep(0.2)
                    continue

            # ждём, когда цена войдёт в полосу ретеста у стены
            if in_retest_band(last):
                if in_band_since == 0.0:
                    in_band_since = time.time()

                # подтверждение отбоя: цена ушла на BOUNCE_TICKS от стены в сторону входа
                if side == "buy":
                    target = wall + BOUNCE_TICKS * tick
                    if last >= target:
                        # короткое подтверждение, что это не тик-шум
                        t_ok = time.time() + BOUNCE_CONFIRM_S
                        while time.time() < t_ok:
                            cur = self.safe_last_price(symbol)
                            if cur < target:
                                break
                            await asyncio.sleep(0.1)
                        else:
                            logger.info(f"[{symbol}] 🔁 Retest подтверждён (BUY). Entry по рынку.")
                            await self.execute_trade_entry(candidate, current_features or {})
                            asyncio.create_task(self._post_entry_failsafe_breakout(candidate))
                            return
                else:
                    target = wall - BOUNCE_TICKS * tick
                    if last <= target:
                        t_ok = time.time() + BOUNCE_CONFIRM_S
                        while time.time() < t_ok:
                            cur = self.safe_last_price(symbol)
                            if cur > target:
                                break
                            await asyncio.sleep(0.1)
                        else:
                            logger.info(f"[{symbol}] 🔁 Retest подтверждён (SELL). Entry по рынку.")
                            await self.execute_trade_entry(candidate, current_features or {})
                            asyncio.create_task(self._post_entry_failsafe_breakout(candidate))
                            return
            else:
                in_band_since = 0.0

            await asyncio.sleep(0.2)

        logger.warning(f"[{symbol}] ⏳ Retest не состоялся в отведённое окно. Сделка пропущена.")
            
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

    def _get_trailing_profile(self, symbol: str) -> dict:
        """
        Возвращает подходящий профиль трейлинга для символа.
        """
        try:
            profs = getattr(config, "TRAILING_PROFILES", {})
            for name, prof in profs.items():
                if symbol in prof.get("symbols", []):
                    return prof
            return getattr(config, "DEFAULT_TRAILING_PROFILE", {})
        except Exception:
            return getattr(config, "DEFAULT_TRAILING_PROFILE", {})


    def _get_trailing_params(self, symbol: str) -> tuple[float, float]:
        """
        Возвращает (start_roi_pct, gap_roi_pct) с приоритетом user_state.json:
        user_state["users"][user_id]["trailing"][mode] -> start_roi_pct/gap_roi_pct
        """
        default_start = 5.0
        default_gap = 2.5

        mode = (self.user_data or {}).get("strategy_mode", "full")
        
        
        start_roi_pct = None
        gap_roi_pct = None

        # 1) user_state.json по user_id
        try:
            uid = str(self.user_id)
            ustate = (self.user_state or {}).get("users", {}).get(uid, {})
            tr = (ustate.get("trailing") or {}).get(mode, {})
            start_roi_pct = tr.get("start_roi_pct")
            gap_roi_pct = tr.get("gap_roi_pct")
        except Exception:
            pass

        # 2) fallback: self.user_data (как и было)
        if start_roi_pct is None:
            start_roi_pct = (self.user_data or {}).get("trailing_start_pct", {}).get(mode) \
                            or (self.user_data or {}).get("trailing_start_pct", {}).get("full")
        if gap_roi_pct is None:
            gap_roi_pct = (self.user_data or {}).get("trailing_gap_pct", {}).get(mode) \
                        or (self.user_data or {}).get("trailing_gap_pct", {}).get("full")

        # 3) дефолты
        start_roi_pct = float(start_roi_pct if start_roi_pct is not None else default_start)
        gap_roi_pct = float(gap_roi_pct if gap_roi_pct is not None else default_gap)

        ml_conf = float(self.position_ml_confidence.get(symbol, 0.5))
        # базово было: start=5.0, gap=2.5 (ROI в % с учетом плеча)
        base_start, base_gap = 5.0, 2.5
        # при сильных (ml→0.9): позже включаем, шире gap; при слабых — наоборот
        start_roi_pct = base_start + 6.0 * (ml_conf - 0.5)   # ~2.0..8.0
        gap_roi_pct   = base_gap   + 2.0 * (ml_conf - 0.5)   # ~1.5..3.5

        return float(start_roi_pct), float(gap_roi_pct)

        #return start_roi_pct, gap_roi_pct

        
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

    async def place_unified_order(self, symbol: str, side: str, qty: float | Decimal, order_type: str, **kwargs):
        cid = kwargs.get("cid") or utils.new_cid()
        comment = kwargs.get("comment", "")
        if self.mode == "demo":
            pos_idx = 0
        elif self.position_mode == 0:
            pos_idx = 0
        else:
            pos_idx = 1 if side == "Buy" else 2
        qty_decimal = qty if isinstance(qty, Decimal) else Decimal(str(qty))
        qty_str = format(qty_decimal.normalize(), "f")
        params = {
            "category":"linear", "symbol":symbol, "side":side, "orderType":order_type,
            "qty": qty_str, "timeInForce":"GTC",
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

    def _snapshot_open_positions(self) -> None:
        try:
            payload = []
            for symbol, pos in self.open_positions.items():
                entry: Dict[str, Any] = {
                    "symbol": symbol,
                    "side": pos.get("side"),
                    "volume": utils.safe_to_float(pos.get("volume")),
                    "avg_price": utils.safe_to_float(pos.get("avg_price")),
                    "leverage": utils.safe_to_float(pos.get("leverage")),
                    "source": pos.get("source"),
                }
                if pos.get("manual_mode"):
                    entry["manual_mode"] = True
                if pos.get("adopted_position"):
                    entry["adopted"] = True
                stop = utils.safe_to_float(pos.get("last_stop_price") or self.last_stop_price.get(symbol))
                if stop:
                    entry["last_stop_price"] = stop
                current_price = self.safe_last_price(symbol)
                if current_price:
                    entry["current_price"] = current_price
                avg_price = entry.get("avg_price") or 0.0
                side = str(entry.get("side") or "").lower()
                if avg_price and current_price:
                    if side == "buy":
                        pnl_pct = (current_price / avg_price - 1.0) * 100.0
                    else:
                        pnl_pct = (avg_price / current_price - 1.0) * 100.0
                    if math.isfinite(pnl_pct):
                        entry["pnl_pct_est"] = round(pnl_pct, 3)
                opened_at = pos.get("open_ts") or pos.get("opened_at")
                if isinstance(opened_at, (int, float)):
                    entry["opened_at"] = dt.datetime.utcfromtimestamp(opened_at).isoformat()
                elif isinstance(opened_at, str):
                    entry["opened_at"] = opened_at
                payload.append(entry)
            path = config.OPEN_POS_JSON
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fp:
                json.dump({str(self.user_id): payload}, fp, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.debug(f"[snapshot] Не удалось сохранить open_positions.json: {exc}")

    def _snapshot_wallet_state(self, wallet_data: dict) -> None:
        try:
            if not isinstance(wallet_data, dict):
                return
            data = dict(wallet_data)
            data.setdefault("timestamp", dt.datetime.utcnow().isoformat())
            payload = {str(self.user_id): data}
            path = config.WALLET_JSON
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.debug(f"[snapshot] Не удалось сохранить wallet_state.json: {exc}")

    async def update_open_positions(self):
        try:
            response = await asyncio.to_thread(lambda: self.session.get_positions(category="linear", settleCoin="USDT"))
            if response.get("retCode") != 0:
                raise ConnectionError(f"API Error: {response.get('retMsg')}")
            
            live_positions = {p["symbol"]: p for p in response.get("result", {}).get("list", []) if utils.safe_to_float(p.get("size", 0)) > 0}
            
            async with self.position_lock:
                for symbol, pos_data in live_positions.items():
                    if symbol not in self.open_positions:
                        logger.warning(f"[SYNC] Обнаружена существующая позиция: {symbol}. Адаптация.")
                        side = pos_data.get("side", "")
                        existing_sl = 0.0
                        for sl_key in ("stopLoss", "stopLossPrice", "stop_loss", "stop_loss_price", "sl"):
                            existing_sl = utils.safe_to_float(pos_data.get(sl_key))
                            if existing_sl > 0:
                                break
                        self.open_positions[symbol] = {
                            "avg_price": utils.safe_to_float(pos_data.get("avgPrice")), "side": side,
                            "volume": utils.safe_to_float(pos_data.get("size")), "leverage": utils.safe_to_float(pos_data.get("leverage", "1")),
                            "source": "adopted", "comment": "Adopted on startup.",
                            "trailing_mode": self.user_data.get("trailing_mode") or config.ACTIVE_TRAILING_MODE,
                            "initial_stop_set": existing_sl > 0,
                            "adopted_position": True,
                            "adopted_existing_stop": existing_sl > 0,
                            "last_stop_price": existing_sl if existing_sl > 0 else None,
                            "manual_mode": True,
                        }
                        if existing_sl > 0:
                            self.last_stop_price[symbol] = existing_sl
                            self.last_sent_stop_price[symbol] = existing_sl
                            logger.info(f"[SYNC] {symbol}: обнаружен существующий стоп {existing_sl:.6f}, сохраняю без изменений.")
                        else:
                            logger.info(f"[SYNC] {symbol}: адаптирован без стопа. Режим управления вручную, стоп не устанавливается.")
                        logger.info(f"[SYNC] {symbol}: переведён в ручной режим сопровождения.")
                        self._snapshot_open_positions()
                        if symbol not in self.watch_tasks and not self.open_positions[symbol].get("manual_mode"):
                            task = asyncio.create_task(self.manage_open_position(symbol))
                            self.watch_tasks[symbol] = task

                for symbol in list(self.open_positions.keys()):
                    if symbol not in live_positions:
                        pos = self.open_positions.get(symbol)
                        if not pos: continue

                        if pos.get('is_closing'):
                            logger.debug(f"[SYNC] Позиция {symbol} обрабатывается execution handler. Пропуск.")
                            continue 

                        if symbol in self.recently_closed_pnl_cache:
                            logger.debug(f"[SYNC] Закрытие {symbol} уже было корректно обработано. Очистка.")
                            self._purge_symbol_state(symbol)
                            continue

                        pos = self.open_positions.get(symbol)
                        if not pos: continue

                        logger.warning(f"[SYNC] Обнаружено ВНЕШНЕЕ закрытие позиции {symbol}. Запускаю расчет PnL...")
                        
                        last_price = 0.0
                        price_source = "N/A"
                        
                        if ticker_info := self.shared_ws.ticker_data.get(symbol, {}):
                            if price := utils.safe_to_float(ticker_info.get("lastPrice")):
                                last_price, price_source = price, "ticker"
                        
                        if not last_price > 0 and (candles := self.shared_ws.candles_data.get(symbol)):
                            if price := utils.safe_to_float(candles[-1].get("closePrice")):
                                last_price, price_source = price, "last_candle"

                        if not last_price > 0:
                            try:
                                resp = await asyncio.to_thread(lambda: self.session.get_public_trade_history(category="linear", symbol=symbol, limit=1))
                                if resp and resp.get("retCode") == 0 and (trade_list := resp.get("result", {}).get("list", [])):
                                    if price := utils.safe_to_float(trade_list[0].get("p")):
                                        last_price, price_source = price, "api_trade"
                            except Exception as e:
                                logger.warning(f"[{symbol}] Ошибка при API-запросе истории сделок: {e}")

                        entry_price = self._resolve_avg_price(symbol, pos)

                        if entry_price > 0 and last_price > 0:
                            pos_volume = utils.safe_to_float(pos.get("volume", 0))
                            pnl_usdt = utils.calc_pnl(pos.get("side", "Buy"), entry_price, last_price, pos_volume)
                            leverage = utils.safe_to_float(pos.get("leverage", 1.0)) or 1.0
                            position_margin = (entry_price * pos_volume) / leverage
                            pnl_pct = (pnl_usdt / position_margin) * 100 if position_margin > 0 else 0.0

                            await self.log_trade(
                                symbol=symbol, side=pos['side'], avg_price=last_price, 
                                volume=pos_volume, action="close", result="closed_by_sync", 
                                pnl_usdt=pnl_usdt, pnl_pct=pnl_pct, 
                                comment=f"Position closed externally. PnL calculated via {price_source}.", 
                                source=pos.get("source", "unknown")
                            )
                            self._register_policy_outcome(symbol, pnl_pct)
                            signal_id = pos.get("signal_id")
                            if signal_id:
                                await self.finalize_dom_signal(signal_id, {
                                    "exit_price": last_price,
                                    "pnl_usdt": pnl_usdt,
                                    "pnl_pct": pnl_pct,
                                    "result": "closed_by_sync",
                                    "close_ts": time.time(),
                                    "symbol": symbol,
                                    "price_source": price_source
                                })
                        else:
                            logger.error(f"[{symbol}] Не удалось рассчитать PnL для внешнего закрытия (нет данных о цене).")
                            await self.log_trade(
                                symbol=symbol, side=pos['side'], avg_price=0, 
                                volume=pos.get("volume", 0), action="close", result="closed_by_sync (pnl_error)", 
                                comment="PNL calculation failed (missing price data)", source=pos.get("source", "unknown")
                            )
                            signal_id = pos.get("signal_id")
                            if signal_id:
                                await self.finalize_dom_signal(signal_id, {
                                    "exit_price": last_price if last_price > 0 else None,
                                    "pnl_usdt": None,
                                    "pnl_pct": None,
                                    "result": "closed_by_sync_error",
                                    "close_ts": time.time(),
                                    "symbol": symbol
                                })
                        self._last_close_ts[symbol] = time.time()
                        self._purge_symbol_state(symbol)
            self._snapshot_open_positions()
        except Exception as e:
            logger.error(f"Критическая ошибка в цикле синхронизации позиций: {e}", exc_info=True)

    async def handle_execution(self, msg: dict):
        for exec_data in msg.get("data", []):
            symbol = exec_data.get("symbol")
            if not symbol:
                continue

            exec_price = utils.safe_to_float(exec_data.get("execPrice"))
            exec_side = exec_data.get("side")
            exec_qty = utils.safe_to_float(exec_data.get("execQty"))

            async with self.position_lock:
                pos = self.open_positions.get(symbol)

                scale_marker = self.pending_scale_ins.get(symbol)

                # Обработка усреднения
                if pos and not pos.get("is_opening") and exec_side == pos.get("side"):
                    if not scale_marker:
                        logger.debug(f"[{symbol}] Игнорируем однонаправленное исполнение без маркера усреднения.")
                        continue

                    marker_link = scale_marker.get("order_link_id")
                    incoming_link = exec_data.get("orderLinkId")
                    if marker_link and incoming_link and incoming_link != marker_link:
                        logger.debug(
                            f"[{symbol}] Исполнение orderLinkId={incoming_link} не совпало с ожидаемым усреднением {marker_link}. Пропуск."
                        )
                        continue

                    if time.time() - scale_marker.get("ts", 0) > 180:
                        logger.warning(f"[{symbol}] Просроченный маркер усреднения → проигнорировано.")
                        self.pending_scale_ins.pop(symbol, None)
                        continue

                    if pos.get("trailing_activated"):
                        logger.warning(f"⛔️ [{symbol}] Усреднение после активации трейлинга проигнорировано.")
                        continue

                    logger.info(f"🎯 [{symbol}] Обнаружено исполнение усредняющего ордера.")
                    old_size = utils.safe_to_float(pos.get("volume", 0))
                    old_avg_price = utils.safe_to_float(pos.get("avg_price", 0))

                    new_size = old_size + exec_qty
                    new_avg_price = ((old_avg_price * old_size) + (exec_price * exec_qty)) / new_size if new_size > 0 else exec_price

                    pos["volume"] = new_size
                    pos["avg_price"] = new_avg_price
                    self.averaging_orders_count[symbol] += 1

                    logger.info(f"🎯 [{symbol}] Позиция увеличена. Ср. цена: {new_avg_price:.6f}, Объём: {new_size}")

                    # Отменяем текущий SL и сразу ставим новый по ATR
                    await self.set_or_amend_stop_loss(0, symbol=symbol, cancel_only=True)
                    logger.info(f"[{symbol}] Текущий стоп-ордер отменен для переустановки.")
                    try:
                        await self._set_initial_stop_loss(symbol, pos, force=True)
                    except Exception as _e_avg:
                        logger.warning(f"[{symbol}] Не удалось переустановить стоп после усреднения: {_e_avg}")
                    if utils.safe_to_float(exec_data.get("leavesQty", 0)) <= 0:
                        self.pending_scale_ins.pop(symbol, None)
                    continue

                if pos and pos.get("is_opening") and exec_data.get("side") == pos.get("side"):
                    if exec_price > 0:
                        pos["avg_price"] = exec_price
                        meta_payload = self.pending_position_meta.pop(symbol, None)
                        if meta_payload:
                            pos.update(meta_payload)
                        pos["open_ts"] = time.time()
                        source_text = self.pending_strategy_comments.pop(symbol, "unknown")
                        pos["source"] = source_text
                        pos["comment"] = f"Strategy: {source_text}"
                        sl_base_key = f"{symbol}_sl_base"
                        if sl_base_key in self.pending_strategy_comments:
                            sl_base = self.pending_strategy_comments.pop(sl_base_key)
                            pos["sl_base_price"] = sl_base
                            logger.info(f"[{symbol}] База для стопа {sl_base} успешно перенесена в данные позиции.")
                        pos.pop("is_opening")
                        policy_record = self.pending_policy_records.pop(symbol, None)
                        if policy_record:
                            policy_record["open_ts"] = pos.get("open_ts")
                            self.learning_positions[symbol] = policy_record
                            if policy_record.get("score") is not None:
                                pos["ml_policy_score"] = policy_record.get("score")
                        logger.info(f"[EXECUTION_OPEN] {pos['side']} {symbol} {pos['volume']:.3f} @ {exec_price:.6f}. Source: '{pos['source']}'")
                        await self.log_trade(symbol=symbol, side=pos['side'], avg_price=exec_price, volume=exec_qty, action="open", result="opened", comment=pos['comment'], source=pos['source'])
                        signal_id = pos.get("signal_id") or self.pending_signal_ids.pop(symbol, None)
                        if signal_id:
                            pos["signal_id"] = signal_id
                            self._update_dom_signal_entry(
                                signal_id,
                                entry_price=exec_price,
                                entry_volume=pos.get("volume", exec_qty),
                                position_open_ts=time.time()
                            )
                        self._snapshot_open_positions()
                        if symbol not in self.watch_tasks:
                            task = asyncio.create_task(self.manage_open_position(symbol))
                            self.watch_tasks[symbol] = task
                    continue

                if pos and exec_data.get("side") != pos.get("side"):
                    pos['is_closing'] = True
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
                        self.recently_closed_pnl_cache[symbol] = {"pnl_usdt": pnl_usdt, "pnl_pct": pnl_pct, "close_price": exit_price, "timestamp": time.time()}
                        await self.log_trade(symbol=symbol, side=pos['side'], avg_price=exit_price, volume=pos_volume, action="close", result="closed_by_execution", pnl_usdt=pnl_usdt, pnl_pct=pnl_pct, comment=pos.get('comment'), source=pos.get("source", "unknown"))
                        self._register_policy_outcome(symbol, pnl_pct)
                        signal_id = pos.get("signal_id")
                        if signal_id:
                            await self.finalize_dom_signal(signal_id, {
                                "exit_price": exit_price,
                                "pnl_usdt": pnl_usdt,
                                "pnl_pct": pnl_pct,
                                "result": "closed_by_execution",
                                "close_ts": time.time(),
                                "symbol": symbol
                            })
                        self._last_close_ts[symbol] = time.time()
                        self._purge_symbol_state(symbol)

                elif not pos and exec_data.get("execPrice"):
                    self.pending_open_exec[symbol] = {"price": utils.safe_to_float(exec_data.get("execPrice")), "side": exec_data.get("side"), "ts": time.time()}

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
                        "comment": None, "source": "",
                        "trailing_mode": self.user_data.get("trailing_mode") or config.ACTIVE_TRAILING_MODE,
                        "is_opening": True
                    }
                    signal_id = self.pending_signal_ids.pop(symbol, None)
                    if signal_id:
                        self.open_positions[symbol]["signal_id"] = signal_id
                        record = self.dom_signal_store.get(signal_id)
                        if record:
                            record["position_open_ts"] = time.time()
                            record["initial_position_size"] = new_size
                    logger.info(f"[PositionStream] NEW_PRELIMINARY {side} {symbol} {new_size:.3f}")
                    

                    pos = self.open_positions.get(symbol) or {}
                    if "trailing_t0" not in pos:
                        pos["trailing_t0"] = time.time()
                    pos.setdefault("be_armed", False)
                    self.open_positions[symbol] = pos


                    pend = self.pending_open_exec.pop(symbol, None)
                    if pend and pend.get("side") == side:
                        pos = self.open_positions[symbol]
                        pos["avg_price"] = pend["price"]
                        pos.pop("is_opening")
                        meta_payload = self.pending_position_meta.pop(symbol, None)
                        if meta_payload:
                            pos.update(meta_payload)
                        pos["open_ts"] = time.time()
                        
                        source_text = self.pending_strategy_comments.pop(symbol, "adopted_unknown")
                        pos["source"] = source_text
                        pos["comment"] = source_text
                        
                        logger.info(f"[EXECUTION_OPEN][adopted] {side} {symbol} {pos['volume']:.3f} @ {pos['avg_price']:.6f}. Source: '{source_text}'")
                        await self.log_trade(
                            symbol=symbol, side=side, avg_price=pos["avg_price"],
                            volume=new_size, action="open", result="opened(adopted)",
                            comment=pos["comment"], source=pos.get("source")
                        )
                        signal_id = pos.get("signal_id") or self.pending_signal_ids.pop(symbol, None)
                        if signal_id:
                            pos["signal_id"] = signal_id
                            self._update_dom_signal_entry(
                                signal_id,
                                entry_price=pos["avg_price"],
                                entry_volume=new_size,
                                position_open_ts=time.time(),
                                adopted=True
                            )
                        if symbol not in self.watch_tasks:
                            task = asyncio.create_task(self.manage_open_position(symbol))
                            self.watch_tasks[symbol] = task
                
                elif symbol in self.open_positions and new_size == 0:
                    logger.debug(f"[PositionStream] {symbol} size=0. Закрытие будет обработано execution handler.")

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
            
            ping_interval = 30
            ping_timeout = 20
            
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
            
        except asyncio.CancelledError:
            logger.info(f"Private WS task для user {self.user_id} отменен.")
            if self.ws_private: self.ws_private.exit()
        except Exception as e:
            logger.error(f"Критическая ошибка при запуске Private WS для user {self.user_id}: {e}", exc_info=True)
            if self.ws_private: self.ws_private.exit()

    async def stop(self):
        logger.info(f"Остановка бота для пользователя {self.user_id}...")
        if self.position_security_task and not self.position_security_task.done():
            self.position_security_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.position_security_task
        await self.secure_open_positions(reason="Shutdown flatten", force_flatten=True, sync_live=True)
        logger.info("🧠 [Память Стен] Финальное сохранение данных перед выходом...")
        self._save_wall_memory()
        if self._wall_memory_save_task and not self._wall_memory_save_task.done():
            self._wall_memory_save_task.cancel()
        if hasattr(self, 'ws_private') and self.ws_private:
            self.ws_private.exit()
        for symbol in list(self.watch_tasks.keys()):
            task = self.watch_tasks.pop(symbol, None)
            if task and not task.done():
                task.cancel()
        logger.info(f"Бот для пользователя {self.user_id} остановлен.")

    # async def route_private_message(self, msg):
    #     topic = (msg.get("topic") or "").lower()
    #     if "position" in topic:
    #         await self.handle_position_update(msg)
    #     elif "execution" in topic:
    #         await self.handle_execution(msg)

    async def route_private_message(self, msg: dict):
        # --- НАЧАЛО НОВОГО БЛОКА ---
        # Проверяем, является ли это сообщение ответом на наш запрос
        req_id = msg.get("req_id") or msg.get("op_id") # Bybit может использовать разные ключи
        if req_id and req_id in self.ws_request_futures:
            future = self.ws_request_futures.get(req_id)
            if future and not future.done():
                # Проверяем, успешный ли ответ
                if msg.get("success", False) or msg.get("ret_msg", "") == "OK":
                    future.set_result(msg)
                else:
                    # Если неуспешный, передаем ошибку
                    error = InvalidRequestError(f"WS request failed: {msg.get('ret_msg', 'Unknown error')}")
                    future.set_exception(error)
            return # Больше ничего с этим сообщением не делаем
        # --- КОНЕЦ НОВОГО БЛОКА ---

        topic = (msg.get("topic") or "").lower()
        if "position" in topic:
            await self.handle_position_update(msg)
        elif "execution" in topic:
            await self.handle_execution(msg)




    # --- ИСПРАВЛЕННАЯ ВЕРСИЯ ---
    # Переписано на систему очков для большей гибкости.
# --- ИСПРАВЛЕННАЯ ВЕРСИЯ V10 ---
    # Теперь рейтинг стены участвует в расчете очков.
    async def _hunt_reversal(self, candidate: dict, features: dict, signal_key: tuple):
        symbol, side = candidate["symbol"], candidate["side"]
        # Получаем рейтинг стены из кандидата
        wall_rating = candidate.get("wall_rating", 0)
        
        hunt_window_sec = self.user_data.get("DOM_HUNT_WINDOW_SEC", 60)
        peak_lock_sec = self.user_data.get("DOM_PEAK_LOCK_SEC", 15)
        score_threshold = 70  # Порог для входа (из 100+)
        hold_ratio = utils.safe_to_float(candidate.get("wall_hold_ratio"), 1.0)

        effective_threshold = score_threshold
        if wall_rating >= 200:
            effective_threshold -= 8
        elif wall_rating >= 120:
            effective_threshold -= 5
        elif wall_rating >= 80:
            effective_threshold -= 3
        effective_threshold = max(55, effective_threshold)

        extreme_price = features.get('price')
        if not extreme_price:
            logger.error(f"💥 [{symbol}/{side}] 'Охотник' не получил начальную цену. Отмена.")
            self.active_signals.discard(signal_key); return
        logger.info(
            f"🏹 [{symbol}] Охотник V10 (Scoring+Rating) активирован. "
            f"Стена R:{wall_rating}, hold:{hold_ratio:.2f}. Порог: {effective_threshold} (base {score_threshold})."
        )
        start_time, last_peak_update_time = time.time(), time.time()

        while time.time() - start_time < hunt_window_sec:
            try:
                await asyncio.sleep(2) # Чуть ускорим проверку
                current_features = await self.extract_realtime_features(symbol)
                if not current_features: continue
                
                current_price = current_features.get('price', 0.0)
                
                # 1. Отслеживание пика/дна (цена идет ПРОТИВ нас)
                if (side == 'Sell' and current_price > extreme_price) or \
                   (side == 'Buy' and current_price < extreme_price):
                    extreme_price = current_price
                    last_peak_update_time = time.time()
                    # Пока цена обновляет экстремумы, мы не входим, ждем остановки
                    continue

                # 2. Ожидание микро-стабилизации после пика
                if time.time() - last_peak_update_time < peak_lock_sec:
                    continue

                # 3. Расчет очков
                score = 0
                reasons = []

                # Баллы за историческую силу стены
                rating_score = 0
                if wall_rating >= 400:
                    rating_score = 40
                elif wall_rating >= 250:
                    rating_score = 34
                elif wall_rating >= 150:
                    rating_score = 28
                elif wall_rating >= 90:
                    rating_score = 22
                elif wall_rating >= 40:
                    rating_score = 15
                elif wall_rating >= 15:
                    rating_score = 8
                if rating_score:
                    score += rating_score
                    reasons.append(f"WallRating+{rating_score}(R:{wall_rating})")

                if hold_ratio >= 0.9:
                    score += 10
                    reasons.append(f"Hold_Strong({hold_ratio:.2f})")
                elif hold_ratio >= 0.75:
                    score += 7
                    reasons.append(f"Hold_Solid({hold_ratio:.2f})")
                elif hold_ratio >= 0.6:
                    score += 3
                    reasons.append(f"Hold_Base({hold_ratio:.2f})")

                # Улика 1: Откат от экстремума (до 30 очков)
                # Считаем откат от найденного extreme_price
                dist = abs(current_price - extreme_price)
                pullback_pct = (dist / extreme_price * 100) if extreme_price > 0 else 0
                
                # Начисляем очки, если откат есть, но не слишком большой (не пропустили движение)
                if 0.15 < pullback_pct < 1.5:
                    pb_score = min(30, int(pullback_pct * 40)) # ~0.75% отката = макс 30 очков
                    score += pb_score
                    reasons.append(f"Pullback({pullback_pct:.2f}%)")

                # Улика 2: Дивергенция по CVD (фиксировано 30 очков)
                cvd_1m = current_features.get('CVD1m', 0.0)
                # Для Sell (шорт) нужен отрицательный CVD (продажи по рынку), и наоборот
                if (side == 'Sell' and cvd_1m < 0) or (side == 'Buy' and cvd_1m > 0):
                    cvd_base = 28 if wall_rating < 150 else 33
                    score += cvd_base
                    reasons.append(f"CVD_Confirm({cvd_1m:.0f})")

                # Улика 3: Истощение объема (до 20 очков)
                vol_1m = current_features.get('vol1m', 0.0)
                avg_vol_30m = current_features.get('avgVol30m', 1.0)
                if avg_vol_30m > 0 and vol_1m < avg_vol_30m:
                    volume_ratio = vol_1m / avg_vol_30m
                    vol_score = int((1 - volume_ratio) * 20)
                    score += vol_score
                    reasons.append(f"Vol_Low(x{volume_ratio:.1f})")

                elapsed = time.time() - start_time
                if wall_rating >= 150:
                    patience_ratio = max(0.0, elapsed - hunt_window_sec * 0.5) / max(1.0, hunt_window_sec * 0.5)
                    patience_bonus = min(8, int(patience_ratio * 12))
                    if patience_bonus > 0:
                        score += patience_bonus
                        reasons.append(f"Patience+{patience_bonus}")

                # 4. Принятие решения
                target_threshold = effective_threshold
                if score >= target_threshold:
                    logger.warning(f"✅ [ОХОТНИК] {symbol}/{side}: Набрано {score}/{target_threshold}! Вход. (R:{wall_rating}). Причины: {', '.join(reasons)}")
                    # Базу для стопа ставим чуть за экстремумом
                    candidate['stop_loss_price_base'] = extreme_price
                    await self.execute_trade_entry(candidate, current_features)
                    self.active_signals.discard(signal_key)
                    return
                else:
                    # Логируем прогресс только если рейтинг стены высокий, чтобы не спамить
                    if wall_rating > 50 and time.time() % 10 < 2.5:
                        logger.debug(f"🏹 [{symbol}] R:{wall_rating}. Счет: {score}/{effective_threshold}. Ждем {side}. {reasons}")

            except Exception as e:
                logger.error(f"💥 [ОХОТНИК] {symbol}/{side}: Ошибка: {e}", exc_info=True)
                break
        
        # Если вышли по тайм-ауту
        if wall_rating > 100:
            logger.info(f"⏳ [ОХОТНИК] {symbol}/{side}: Тайм-аут на сильной стене (R:{wall_rating}). Разворота не было.")
        self.active_signals.discard(signal_key)
        

    async def _hunter_precheck(self, symbol: str, side: str) -> bool:
        """
        Пер-символьный кулдаун + предфильтры ликвидности/ADX/спред до запуска окна охотника.
        """

        hcfg = getattr(config, "BREAKOUT_HUNTER", {})
        scfg = getattr(config, "SYMBOL_FILTERS", {})

        # чёрный список
        if symbol in set(scfg.get("BLACKLIST", [])):
            return False

        # пер-символьный кулдаун на старты охотника
        if not hasattr(self, "_hunter_last_start_ts"):
            self._hunter_last_start_ts = {}
        last = self._hunter_last_start_ts.get(symbol, 0.0)
        now  = time.time()
        min_cd = float(hcfg.get("COOLDOWN_AFTER_CANCEL_SEC", 30.0))
        if now - last < min_cd:
            return False

        # проверка спреда в тиках
        tick = float(self.price_tick_map.get(symbol, 0.0) or 0.0) or 1e-6
        best_bid = utils.safe_to_float(getattr(self, "best_bid_map", {}).get(symbol, 0.0))
        best_ask = utils.safe_to_float(getattr(self, "best_ask_map", {}).get(symbol, 0.0))
        if best_bid > 0.0 and best_ask > 0.0 and best_ask >= best_bid:
            spread_ticks = int(round((best_ask - best_bid) / tick))
            if spread_ticks > int(hcfg.get("MAX_SPREAD_TICKS", 4)):
                return False

        # MIN_ADX
        features = await self.extract_realtime_features(symbol)
        if features:
            adx = float(features.get("adx14", 0.0))
            if adx < float(hcfg.get("MIN_ADX", 18.0)):
                return False

        # прошли все фильтры — фиксируем старт
        self._hunter_last_start_ts[symbol] = now
        return True

    async def _golden_precheck(self, symbol: str, side: str) -> bool:
        settings = getattr(self, "golden_settings", getattr(config, "GOLDEN_HUNTER", {}))

        if not hasattr(self, "_golden_last_start_ts"):
            self._golden_last_start_ts = {}

        last = self._golden_last_start_ts.get(symbol, 0.0)
        now = time.time()
        min_cd = float(settings.get("COOLDOWN_SEC", getattr(self, "golden_hunter_cooldown_sec", 120.0)))
        if now - last < min_cd:
            return False

        min_turnover = getattr(self, "golden_min_turnover", 0.0)
        if min_turnover > 0:
            turnover = self._get_symbol_turnover(symbol)
            if turnover and turnover < min_turnover:
                return False

        tick = float(self.price_tick_map.get(symbol, 0.0) or 1e-6)
        best_bid = utils.safe_to_float(getattr(self, "best_bid_map", {}).get(symbol, 0.0))
        best_ask = utils.safe_to_float(getattr(self, "best_ask_map", {}).get(symbol, 0.0))
        max_spread_ticks = int(settings.get("MAX_SPREAD_TICKS", getattr(self, "golden_max_spread_ticks", 6)))
        if best_bid > 0.0 and best_ask > 0.0 and best_ask >= best_bid:
            spread_ticks = int(round((best_ask - best_bid) / tick))
            if spread_ticks > max_spread_ticks:
                return False

        min_adx = float(settings.get("MIN_ADX", getattr(self, "golden_min_adx", 0.0)))
        min_avg_vol = getattr(self, "golden_min_avg_vol", 0.0)
        require_features = min_adx > 0 or min_avg_vol > 0
        features = await self.extract_realtime_features(symbol) if require_features else None
        if min_adx > 0 and features:
            adx = float(features.get("adx14", 0.0))
            if adx < min_adx:
                return False
        if min_avg_vol > 0:
            if not features:
                return False
            avg_vol = utils.safe_to_float(
                features.get("avg_volume_prev_4m")
                or features.get("GS_vol4m")
                or 0.0
            )
            if avg_vol <= 0 or avg_vol < min_avg_vol:
                return False

        self._golden_last_start_ts[symbol] = now
        return True

    # --- ИСПРАВЛЕННАЯ ВЕРСИЯ ---
    # Упрощена и сделана более читаемой.
    async def _initiate_hunt(self, candidate: dict, features: dict, signal_key: tuple):
        """
        Маршрутизирует сигналы в соответствующие тактические группы ('Охотники').
        """
        source = (candidate.get("source") or "").lower()
        symbol = candidate.get("symbol")

        # Нормализуем сторону сделки (надёжно, с фолбэками)
        side_raw = (
            candidate.get("side")
            or candidate.get("entry_side")
            or candidate.get("direction")
            or (features or {}).get("side")
            or ""
        )
        side = str(side_raw).strip().lower()
        if side not in ("buy", "sell"):
            # пытаемся угадать из текстовых полей
            text_blob = " ".join([
                source,
                str(candidate.get("signal") or ""),
                str(candidate.get("label") or ""),
            ]).lower()
            if "sell" in text_blob:
                side = "sell"
            elif "buy" in text_blob:
                side = "buy"
            else:
                # безопасный дефолт (только чтобы не падать в логах/чеке)
                side = "buy"

        # Группа "Охотников за разворотом"
        if any(k in source for k in ("fade", "squeeze", "liquidation")):
            logger.info(f"🎯 [{symbol}] Запуск охотника на РАЗВОРОТ для '{source}'.")
            asyncio.create_task(self._hunt_reversal(candidate, features, signal_key))
            return

        candidate["side"] = side.capitalize()

        # Golden Setup (быстрый пробой)
        if "golden_setup" in source:
            if candidate.get("direct_entry"):
                ok_guard, reason = await self._entry_guard(symbol, side, features, candidate)
                if not ok_guard:
                    logger.info(f"[Охотник][GOLDEN] {symbol}/{side}: direct-entry отклонён guard'ом ({reason}).")
                    self.active_signals.discard(signal_key)
                    return
                logger.info(f"⚡️ [{symbol}] GOLDEN DIRECT ENTRY | side={side} | score={candidate.get('score_total')} | ml={candidate.get('ml_conf'):.2f}")
                await self.execute_trade_entry(candidate, features)
                self.active_signals.discard(signal_key)
                return
            if not await self._golden_precheck(symbol, side):
                logger.debug(f"[Охотник][GOLDEN] {symbol}/{side}: пропущен предфильтром.")
                self.active_signals.discard(signal_key)
                return
            logger.info(f"🎯 [{symbol}] Запуск охотника на ПРОБОЙ (Golden).")
            asyncio.create_task(self._hunt_golden_breakout(candidate, features, signal_key))
            return

        # Группа "Охотников за пробоем" (DOM)
        if "breakout" in source:
            # предчек (кулдауны/ликвидность/ADX/спред) — нужен определённый side (для логов и потенциальных хелперов)
            if not await self._hunter_precheck(symbol, side):
                logger.debug(f"[Охотник] {symbol}/{side}: пропущен предфильтром (кулдаун/ликвидность/ADX/спред).")
                return
            logger.info(f"🎯 [{symbol}] Запуск охотника на ПРОБОЙ для '{source}'.")
            asyncio.create_task(self._hunt_golden_breakout(candidate, features, signal_key))
            return

        # Если стратегия не требует тактического ожидания
        logger.warning(f"[{symbol}] Для источника '{source}' не найден 'Охотник'. Исполнение сразу.")
        await self.execute_trade_entry(candidate, features)
        self.active_signals.discard(signal_key)

    # --- ИСПРАВЛЕННАЯ ВЕРСИЯ ---
    # Это ваша архитектура V4, которая обходит AI для принятия решений,
    # что является разумным шагом при отладке основной логики.
    # Я оставил эту версию как основную.
    async def _process_signal(self, candidate: dict, features: dict, signal_key: tuple):
        """
        [АРХИТЕКТУРА V4] Проверяет базовые условия и передает сигнал
        в соответствующую тактическую группу ("Охотника").
        """
        try:
            if self.pattern_signals_enabled and features:
                asyncio.create_task(
                    self._emit_pattern_signal(dict(candidate), dict(features))
                )
            # 1. Финальная проверка перед передачей в тактическую группу
            # (Guard вызывается уже внутри execute_trade_entry, здесь его дублировать не нужно)

            # 2. Передача "Охотнику"
            await self._initiate_hunt(candidate, features, signal_key)
            
        except Exception as e:
            logger.error(f"Критическая ошибка в _process_signal для {signal_key}: {e}", exc_info=True)
            self.active_signals.discard(signal_key)

    async def _emit_pattern_signal(self, candidate: Dict[str, Any], features: Dict[str, Any]):
        """
        Формирует и отправляет пользователю независимый сигнал от ML-модели.
        """
        try:
            if not self.pattern_signals_enabled:
                return
            if not isinstance(features, dict) or not features:
                return
            if self.ml_inferencer is None:
                return

            symbol = str(candidate.get("symbol") or "").upper()
            if not symbol:
                return

            now = time.time()
            last_ts = self._last_pattern_signal_ts.get(symbol, 0.0)
            if now - last_ts < self.pattern_signal_cooldown_sec:
                return

            feature_vector = [utils.safe_to_float(features.get(key), 0.0) for key in config.FEATURE_KEYS]
            inference_input = np.asarray([feature_vector], dtype=np.float32)
            try:
                raw_pred = self.ml_inferencer.infer(inference_input)
            except Exception as inf_err:
                logger.error(f"[PatternSignal] {symbol}: инференс не удался: {inf_err}", exc_info=True)
                return

            score = float(np.asarray(raw_pred).reshape(-1)[0])
            if not math.isfinite(score):
                score = 0.0
            probability = 0.5 * (1.0 + math.tanh(score / 2.0))
            probability = float(np.clip(probability, 0.0, 1.0))

            tone_line = self._choose_pattern_signal_tone(probability)
            direction_text, direction_emoji = self._pattern_direction(candidate)
            pattern_name = self._describe_pattern_source(candidate)
            metrics_line = self._build_pattern_metric_line(features)
            details_line = self._format_candidate_context(candidate)

            message_parts = [
                tone_line,
                f"{direction_emoji} Появился признак входа в {direction_text} по <b>{symbol}</b> ({pattern_name}).",
                f"🔮 Прогноз на положительный исход: <b>{probability * 100:.2f}%</b>.",
            ]
            chart_url = self._build_chart_url(symbol)
            if chart_url:
                message_parts.append(f"🔗 График: {chart_url}")
            if metrics_line:
                message_parts.append(metrics_line)
            if details_line:
                message_parts.append(details_line)

            text = "\n".join(message_parts)
            await self.notify_user(text)
            self._last_pattern_signal_ts[symbol] = now
            logger.info(f"[PatternSignal] {symbol} {direction_text.upper()} → {probability * 100:.2f}% ({pattern_name}).")
        except Exception as e:
            logger.error(f"[PatternSignal] Ошибка формирования сигнала: {e}", exc_info=True)

    def _choose_pattern_signal_tone(self, probability: float) -> str:
        high_threshold = max(self.pattern_signal_min_confidence, 0.75)
        if probability >= high_threshold:
            return "⚡️ <b>Сильный паттерн</b>"
        if probability >= self.pattern_signal_min_confidence:
            return "📡 <b>Паттерн обнаружен</b>"
        return "🕊 <b>Нужна дополнительная проверка</b>"

    def _pattern_direction(self, candidate: Dict[str, Any]) -> tuple[str, str]:
        side_raw = (
            candidate.get("side")
            or candidate.get("entry_side")
            or candidate.get("direction")
            or ""
        )
        side = str(side_raw).strip().lower()
        if side in ("sell", "short", "bear"):
            return "шорт", "🔴"
        return "лонг", "🟢"

    def _describe_pattern_source(self, candidate: Dict[str, Any]) -> str:
        source = str(candidate.get("source") or "").lower()
        label = str(candidate.get("label") or "").strip()
        if "golden" in source:
            return "Golden Setup"
        if "squeeze" in source:
            return "Squeeze"
        if "liquid" in source or "liq" in source:
            return "Liquidation Pulse"
        if "dom" in source:
            return "Orderbook Pattern"
        if "breakout" in source:
            return "Breakout Hunter"
        if label:
            return label
        original_source = candidate.get("source")
        return str(original_source) if original_source else "Паттерн"

    def _build_pattern_metric_line(self, features: Dict[str, Any]) -> Optional[str]:
        metrics: list[str] = []

        def _fetch(key: str) -> Optional[float]:
            if key not in features:
                return None
            try:
                return float(features[key])
            except (TypeError, ValueError):
                return None

        pct5m = _fetch("pct5m")
        if pct5m is not None:
            metrics.append(f"Δ5м {pct5m:+.2f}%")

        doi5m = _fetch("dOI5m")
        if doi5m is not None:
            metrics.append(f"ΔOI5м {doi5m * 100:+.2f}%")

        adx = _fetch("adx14")
        if adx is not None and adx > 0:
            metrics.append(f"ADX {adx:.1f}")

        rsi = _fetch("rsi14")
        if rsi is not None and rsi > 0:
            metrics.append(f"RSI {rsi:.1f}")

        vol_anom = _fetch("volume_anomaly")
        if vol_anom is not None and vol_anom > 0:
            metrics.append(f"Объём ×{vol_anom:.2f}")

        if not metrics:
            return None

        return "Ключевые факторы: " + " | ".join(metrics[:5])

    def _build_chart_url(self, symbol: str) -> Optional[str]:
        if not symbol:
            return None
        formatted = str(symbol).upper().replace("/", "")
        if not formatted.endswith("USDT"):
            return None
        return f"https://www.bybit.com/trade/usdt/{formatted}"

    def _format_candidate_context(self, candidate: Dict[str, Any]) -> Optional[str]:
        details: list[str] = []

        def _append(label: str, value_key: str, fmt: str = "{:.2f}"):
            if value_key not in candidate:
                return
            try:
                value = float(candidate[value_key])
            except (TypeError, ValueError):
                return
            if "rating" in value_key or "cluster" in value_key:
                fmt_local = "{:.0f}"
            else:
                fmt_local = fmt
            formatted = f"{label} {fmt_local.format(value)}"
            if formatted not in details:
                details.append(formatted)

        _append("z", "wall_zscore")
        _append("rating", "wall_rating")
        _append("кластер", "cluster_size", "{:.0f}")
        _append("мощность", "sq_power")
        _append("мощность", "SQ_power")
        _append("стена", "wall_price", "{:.6f}")
        _append("спред%", "spread_pct")

        if not details:
            return None

        return "Детали паттерна: " + " | ".join(details)

    async def register_dom_signal(self, record: Dict[str, Any]):
        signal_id = record.get("signal_id")
        if not signal_id:
            return
        record.setdefault("signal_ts", time.time())
        self.dom_signal_store[signal_id] = record

    def _update_dom_signal_entry(self, signal_id: str, **kwargs):
        record = self.dom_signal_store.get(signal_id)
        if not record:
            return
        record.update(kwargs)

    async def finalize_dom_signal(self, signal_id: str, outcome: Dict[str, Any]):
        record = self.dom_signal_store.pop(signal_id, None)
        if not record:
            return

        record.update(outcome)
        record.setdefault("close_ts", time.time())
        open_ts = record.get("position_open_ts") or record.get("signal_ts")
        if open_ts:
            record["holding_time_sec"] = max(0.0, record["close_ts"] - open_ts)

        manual_label = outcome.get("label")
        if manual_label is not None:
            record["label"] = int(manual_label)
        else:
            pnl_raw = record.get("pnl_usdt")
            pnl_usdt = utils.safe_to_float(pnl_raw) if pnl_raw is not None else None
            if pnl_usdt is None:
                record["label"] = 0
            else:
                record["label"] = 1 if pnl_usdt >= 0 else 0

        self.training_data.append(record)
        await self._persist_training_sample(record)

    async def _persist_training_sample(self, sample: Dict[str, Any]):
        path = config.TRAINING_DATA_PATH

        async with self.dom_dataset_lock:
            def _write():
                data = []
                try:
                    if path.exists():
                        with open(path, "rb") as fp:
                            data = pickle.load(fp) or []
                except Exception:
                    data = []
                data.append(sample)
                max_len = 5000
                if len(data) > max_len:
                    data = data[-max_len:]
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as fp:
                    pickle.dump(data, fp)

            await asyncio.to_thread(_write)


    async def _hunt_squeeze_reversal(self, candidate: dict, features: dict, signal_key: tuple):
        """
        [ТЕРПЕЛИВЫЙ ОХОТНИК V3.1] Сначала выслеживает истинный пик импульса,
        затем ждет УДЛИНЕННОЙ стабилизации и более весомых подтверждений.
        """
        symbol = candidate["symbol"]
        side = candidate["side"]
        initial_rsi = features.get('rsi14', 50.0)
        
        PEAK_LOCK_IN_DURATION_SEC = 30 
        CONFIRMATION_STRIKES_NEEDED = 3
        ENTRY_SCORE_THRESHOLD = 65
        
        extreme_price = features.get('price', 0.0)
        if extreme_price == 0:
            logger.error(f"💥 [{symbol}/{side}] 'Охотник' не получил начальную цену. Отмена.")
            self.active_signals.discard(signal_key)
            return

        start_time = time.time()
        last_peak_update_time = start_time
        confirmation_strikes = 0
        
        logger.info(f"🏹 [{symbol}] Двухфазный Охотник V3.1 активирован. Начальный экстремум: {extreme_price:.6f}")

        while time.time() - start_time < self.tactical_entry_window_sec:
            try:
                current_features = await self.extract_realtime_features(symbol)
                if not current_features:
                    await asyncio.sleep(self.squeeze_ai_confirm_interval_sec)
                    continue

                current_price = current_features.get('price', 0.0)
                is_new_extreme = False
                if side == 'Sell' and current_price > extreme_price:
                    extreme_price = current_price
                    is_new_extreme = True
                elif side == 'Buy' and current_price < extreme_price:
                    extreme_price = current_price
                    is_new_extreme = True

                if is_new_extreme:
                    logger.debug(f"🏹 [{symbol}] Новый экстремум отслежен: {extreme_price:.6f}. Таймер подтверждения сброшен.")
                    last_peak_update_time = time.time()
                    confirmation_strikes = 0
                    await asyncio.sleep(self.squeeze_ai_confirm_interval_sec)
                    continue

                if time.time() - last_peak_update_time < PEAK_LOCK_IN_DURATION_SEC:
                    logger.debug(f"🏹 [{symbol}] Ожидание стабилизации на пике ({int(time.time() - last_peak_update_time)}/{PEAK_LOCK_IN_DURATION_SEC}с)...")
                    await asyncio.sleep(self.squeeze_ai_confirm_interval_sec)
                    continue
                
                score, reasons = self._calculate_squeeze_reversal_score(
                    side, initial_rsi, extreme_price, current_features
                )
                
                if score >= ENTRY_SCORE_THRESHOLD:
                    confirmation_strikes += 1
                    logger.info(f"🏹 [{symbol}] Подтверждение разворота {confirmation_strikes}/{CONFIRMATION_STRIKES_NEEDED}. Счет: {score}. Причины: {', '.join(reasons)}")
                else:
                    if confirmation_strikes > 0:
                        logger.info(f"🏹 [{symbol}] Условие входа нарушено, сброс счетчика подтверждений.")
                    confirmation_strikes = 0

                if confirmation_strikes >= CONFIRMATION_STRIKES_NEEDED:
                    logger.info(f"✅ [ОХОТНИК] {symbol}/{side}: Цель подтверждена! Вход разрешен.")
                    candidate['stop_loss_price_base'] = extreme_price
                    await self.execute_trade_entry(candidate, current_features)
                    return
                    
                await asyncio.sleep(self.squeeze_ai_confirm_interval_sec)
                
            except Exception as e:
                logger.error(f"💥 [ОХОТНИК] {symbol}/{side}: Критическая ошибка в цикле: {e}", exc_info=True)
                break
                
        logger.warning(f"⏳ [ОХОТНИК] {symbol}/{side}: Окно входа истекло, разворот не подтвержден.")
        signal_id = candidate.get("signal_id")
        if signal_id:
            await self.finalize_dom_signal(signal_id, {
                "result": "no_entry",
                "close_ts": time.time(),
                "label": 0,
                "symbol": symbol
            })
        self.active_signals.discard(signal_key)

    async def _hunt_golden_breakout(self, candidate: dict, features: dict, signal_key: tuple):
            """
            [V3 - "СЛЕДОПЫТ"] Ищет подтверждение пробоя, используя систему очков.
            """
            symbol = candidate["symbol"]
            side = candidate["side"]
            reference_price = features.get("price", 0.0)
            
            cfg = dict(getattr(config, "BREAKOUT_HUNTER", {}))
            if isinstance(self.golden_settings, dict):
                cfg.update(self.golden_settings)

            ml_conf = float(candidate.get("ml_conf", 0.5))
            base_window = int(cfg.get("BREAKOUT_HUNT_WINDOW_SEC", cfg.get("WINDOW_SEC", 10)))
            base_threshold = int(cfg.get("BREAKOUT_CONFIRMATION_SCORE", cfg.get("REQUIRED_SCORE", 70)))

            # линейно двигаем: при 0.9 уверенности – окно короче, порог ниже
            hunt_window_sec = int(max(5, base_window - 6 * (ml_conf - 0.5)))
            score_threshold = int(base_threshold - 20 * (ml_conf - 0.5))
            # защитные клипы
            hunt_window_sec = max(5, min(base_window, hunt_window_sec))
            score_threshold = max(40, min(base_threshold, score_threshold))
            direct_score = candidate.get("score_total")
            if isinstance(direct_score, (int, float)):
                score_threshold = max(35, score_threshold - int(10 * max(0, direct_score - 1)))
            
            start_time = time.time()
            logger.info(f"🏹 [{symbol}] Охотник-Следопыт V3 активирован. Окно: {hunt_window_sec}с, Порог: {score_threshold} очков.")

            while time.time() - start_time < hunt_window_sec:
                try:
                    await asyncio.sleep(2.0)
                    
                    current_features = await self.extract_realtime_features(symbol)
                    if not current_features: continue

                    last_price = current_features.get("price")
                    if not last_price: continue

                    # --- Сбор улик и начисление очков ---
                    score = 0
                    reasons = []

                    # 1. Улика: Цена пробила уровень
                    price_change_pct = ((last_price - reference_price) / reference_price) * 100.0 if reference_price > 0 else 0
                    price_pass = (side == "Buy" and price_change_pct >= cfg.get("PRICE_DELTA_THRESHOLD", 0.2)) or \
                                 (side == "Sell" and price_change_pct <= -cfg.get("PRICE_DELTA_THRESHOLD", 0.2))
                    if price_pass:
                        score += cfg.get("SCORE_PRICE_CONFIRMED", 40)
                        reasons.append(f"Price(Δ{price_change_pct:.2f}%)")

                    # 2. Улика: Поток ордеров (CVD) поддерживает движение
                    cvd_1m = current_features.get("CVD1m", 0)
                    if (side == "Buy" and cvd_1m > 0) or \
                    (side == "Sell" and cvd_1m < 0):
                        score += cfg.get("SCORE_FLOW_CONFIRMED", 40)
                        reasons.append("Flow")

                    # 3. Улика: Есть всплеск объема
                    volume_anomaly = float(current_features.get("volume_anomaly", 1.0) or 1.0)
                    if volume_anomaly > 1.5:
                        score += cfg.get("SCORE_VOLUME_CONFIRMED", 20)
                        reasons.append("Volume")

                    extreme_level = cfg.get("EXTREME_VOLUME_SPIKE", 50.0)
                    if volume_anomaly >= extreme_level:
                        bonus = cfg.get("SCORE_VOLUME_EXTREME", 10)
                        score += bonus
                        reasons.append(f"VolumeExtreme+{bonus}")
                        logger.warning(
                            f"⚡️ [СЛЕДОПЫТ] {symbol}/{side}: экстремальный всплеск объёма x{volume_anomaly:.1f} (порог {extreme_level})."
                        )

                    logger.info(
                        f"[СЛЕДОПЫТ] {symbol}/{side}: score={score}/{score_threshold} | "
                        f"Δprice={price_change_pct:.2f}% | CVD1m={cvd_1m:.2f} | "
                        f"vol_anom={volume_anomaly:.2f} | reasons={','.join(reasons) if reasons else 'none'}"
                        f" | elapsed={time.time()-start_time:.1f}s"
                    )

                    # --- Принятие решения ---
                    flow_support = (side == "Buy" and cvd_1m > 0) or (side == "Sell" and cvd_1m < 0)
                    if score >= score_threshold and price_pass and flow_support:
                        logger.info(
                            f"✅ [СЛЕДОПЫТ] {symbol}/{side}: Достаточно улик! Счет: {score}/{score_threshold}. Причины: {', '.join(reasons)}. Исполнение."
                        )

                        if candidate.get("source") == "mlx_dom_breakout":
                            ok_pref = await self._pre_flight_revalidate_breakout(candidate)
                            if not ok_pref:
                                logger.warning(f"[{symbol}] [ПРЕ-ФЛАЙТ] Breakout отменён: нет приёмки за стеной/дрейф назад.")
                                self.active_signals.discard(signal_key)
                                signal_id = candidate.get("signal_id")
                                if signal_id:
                                    await self.finalize_dom_signal(signal_id, {
                                        "result": "no_entry_prefail",
                                        "close_ts": time.time(),
                                        "label": 0,
                                        "symbol": symbol
                                    })
                                return

                        # --- ФОЛЛБЭК ДЛЯ GOLDEN В МЕЖСЕССИОНКУ ---
                        try:
                            if (candidate.get("source") == "golden_setup") and is_quiet_market_window():
                                fb_cfg = self.user_data.get("dom_squeeze_settings", config.DOM_SQUEEZE_STRATEGY).get("FALLBACK", {})
                                FALLBACK_SEC   = float(fb_cfg.get("GOLDEN_FALLBACK_SEC", 60))
                                MAX_DEV_ATR    = float(fb_cfg.get("MAX_DEV_ATR", 0.25))
                                SIZE_FACTOR    = float(fb_cfg.get("SIZE_FACTOR", 0.5))

                                elapsed = time.time() - start_time
                                last_price = current_features.get("price")
                                atr1 = current_features.get("atr_1m") or current_features.get("atr_5m", 0.0)

                                # Допуск: цена не ушла далеко против референса
                                ok_dev = False
                                if atr1 and reference_price:
                                    if side == "Buy":
                                        ok_dev = last_price >= reference_price - MAX_DEV_ATR * atr1
                                    else:  # Sell
                                        ok_dev = last_price <= reference_price + MAX_DEV_ATR * atr1

                                if elapsed >= FALLBACK_SEC and atr1 and ok_dev:
                                    # Проверка guard и лимита объёма
                                    ok_guard, reason = await self._entry_guard(symbol, side, current_features, candidate)
                                    if not ok_guard:
                                        logger.info(f"[{symbol}] Fallback отклонён guard'ом: {reason}")
                                    else:
                                        volume_to_open = self.POSITION_VOLUME * SIZE_FACTOR
                                        effective_total_vol = await self.get_effective_total_volume()
                                        if effective_total_vol + volume_to_open > self.MAX_TOTAL_VOLUME:
                                            logger.warning(f"[{symbol}] [FALLBACK_REJECT] Превышен лимит общего объёма.")
                                        else:
                                            # Регистрируем pending и исполняем
                                            async with self.pending_orders_lock:
                                                if symbol in self.open_positions:
                                                    logger.warning(f"[{symbol}] [FALLBACK_SKIP] Позиция уже открыта.")
                                                else:
                                                    self.pending_orders[symbol] = volume_to_open
                                                    self.pending_timestamps[symbol] = time.time()
                                                    if signal_key:
                                                        self.pending_signal_ids[symbol] = signal_key[0] if isinstance(signal_key, tuple) else signal_key
                                                    self.pending_strategy_comments[symbol] = "golden_fallback"

                                            try:
                                                qty = await self._calc_qty_from_usd(symbol, volume_to_open)
                                                if qty <= 0:
                                                    raise ValueError("Рассчитан нулевой объём для fallback.")
                                                await self.place_unified_order(
                                                    symbol=symbol,
                                                    side=side,
                                                    qty=qty,
                                                    order_type="Market",
                                                    comment="Golden Fallback (intersession)"
                                                )
                                                self.last_entry_ts[symbol] = time.time()
                                                logger.warning(f"⚡️ [{symbol}] GOLDEN FALLBACK вход выполнен: side={side}, size≈{volume_to_open:.2f} USDT (elapsed={int(elapsed)}s)")
                                                logger.info(
                                                    f"🕒 [{symbol}] Охотник завершён без входа: score<{score_threshold} "
                                                    f"за {int(time.time()-start_time)}с."
                                                )

                                                self.active_signals.discard(signal_key)
                                                return
                                            except Exception as e:
                                                logger.error(f"[{symbol}] Fallback ошибка исполнения: {e}", exc_info=True)
                                                async with self.pending_orders_lock:
                                                    self.pending_orders.pop(symbol, None)
                                                    self.pending_timestamps.pop(symbol, None)
                                                    self.pending_signal_ids.pop(symbol, None)
                        except Exception:
                            logger.exception(f"[{symbol}] Fallback block exception")


                            dcfg = self.user_data.get("dom_squeeze_settings", config.DOM_SQUEEZE_STRATEGY)
                            entry_mode = str(dcfg.get("ENTRY_MODE", "retest")).lower()

                            if entry_mode == "retest":
                                await self._enter_breakout_on_retest(candidate, current_features)
                                self.active_signals.discard(signal_key)
                                return

                        await self.execute_trade_entry(candidate, current_features)

                        if candidate.get("source") == "mlx_dom_breakout":
                            asyncio.create_task(self._post_entry_failsafe_breakout(candidate))
                        
                        self.active_signals.discard(signal_key)
                        return

                except Exception as e:
                    logger.error(f"💥 [СЛЕДОПЫТ] {symbol}/{side}: Критическая ошибка в цикле: {e}", exc_info=True)
                    break
                    
            logger.warning(f"⏳ [СЛЕДОПЫТ] {symbol}/{side}: Окно входа истекло, недостаточно улик для подтверждения пробоя.")
            signal_id = candidate.get("signal_id")
            if signal_id:
                await self.finalize_dom_signal(signal_id, {
                    "result": "no_entry",
                    "close_ts": time.time(),
                    "label": 0,
                    "symbol": symbol
                })
            self.active_signals.discard(signal_key)




    async def execute_priority_trade(self, candidate: dict):
        symbol = candidate.get("symbol")
        side = candidate.get("side")
        source = candidate.get("source", "N/A")
        is_averaging_trade = candidate.get("is_averaging", False)
        log_prefix = "AVERAGING" if is_averaging_trade else "INSIDER"
        if is_averaging_trade:
            logger.warning(f"🎯 [{symbol}] 'УСРЕДНЕНИЕ' АКТИВИРОВАНО! Вход по {side} для улучшения средней цены.")
        else:
            logger.warning(f"⚡️ [{symbol}] '{log_prefix}' АКТИВИРОВАН! Сигнал '{source}'. Немедленное исполнение.")
        async with self.pending_orders_lock:
            if not is_averaging_trade and (symbol in self.open_positions or symbol in self.pending_orders):
                logger.warning(f"[{log_prefix}_SKIP] Позиция по {symbol} уже существует. Вход отменен.")
                return
            volume_to_open = self.POSITION_VOLUME
            signal_id = candidate.get("signal_id")
            effective_total_vol = await self.get_effective_total_volume()
            if effective_total_vol + volume_to_open > self.MAX_TOTAL_VOLUME:
                logger.warning(f"[{log_prefix}_REJECT] Превышен лимит общего объема.")
                return
            self.pending_orders[symbol] = volume_to_open
            self.pending_timestamps[symbol] = time.time()
            if signal_id:
                self.pending_signal_ids[symbol] = signal_id
            if not is_averaging_trade:
                self.pending_strategy_comments[symbol] = source
                if 'stop_loss_price_base' in candidate:
                    sl_base = candidate['stop_loss_price_base']
                    self.pending_strategy_comments[f"{symbol}_sl_base"] = sl_base
                    logger.info(f"[{symbol}] База для стопа {sl_base} сохранена в pending_strategy_comments.")
        cid = utils.new_cid()
        try:
            qty = await self._calc_qty_from_usd(symbol, volume_to_open)
            if qty <= 0: raise ValueError("Рассчитан нулевой объем.")
            if is_averaging_trade:
                self.pending_scale_ins[symbol] = {
                    "order_link_id": cid,
                    "side": side,
                    "ts": time.time(),
                    "target_qty": qty,
                }
            comment_text = f"Averaging Entry" if is_averaging_trade else f"Insider Signal: {source}"
            await self.place_unified_order(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type="Market",
                comment=comment_text,
                cid=cid,
            )
            self.last_entry_ts[symbol] = time.time()
        except Exception as e:
            logger.error(f"[{log_prefix}_CRITICAL] Критическая ошибка при исполнении входа для {symbol}: {e}", exc_info=True)
            async with self.pending_orders_lock:
                self.pending_orders.pop(symbol, None)
                self.pending_timestamps.pop(symbol, None)
            if is_averaging_trade:
                self.pending_scale_ins.pop(symbol, None)

    async def manage_open_position(self, symbol: str):
        logger.info(f"🛡️ [Guardian V14] Активирован для позиции {symbol}.")
        try:
            await asyncio.sleep(1.5)
            pos = self.open_positions.get(symbol)
            if not pos: 
                logger.warning(f"[{symbol}] Guardian не нашел открытую позицию после активации.")
                return
            if pos.get("manual_mode"):
                logger.info(f"🛡️ [{symbol}] Ручной режим сопровождения. Guardian передаёт управление оператору.")
                return
            if pos.get("adopted_position") and pos.get("adopted_existing_stop"):
                pos["initial_stop_set"] = True
                logger.info(f"🛡️ [{symbol}] Адаптация: найден существующий стоп. Пропускаю установку нового SL.")
            else:
                await self._set_initial_stop_loss(symbol, pos)
            if 'dom_squeeze' in pos.get("source", ""):
                if self.user_data.get("dom_squeeze_settings", {}).get("AVERAGING_ENABLED", True):
                    logger.info(f"🛡️ [{symbol}] Активирован режим усреднения.")
                    asyncio.create_task(self._manage_averaging(symbol))
            logger.info(f"🕹️ [{symbol}] Первоначальная настройка завершена. Управление передано тиковому трейлингу 'Трещотка'.")
        except asyncio.CancelledError:
            logger.info(f"[Guardian] Наблюдение за {symbol} отменено.")
        except Exception as e:
            logger.error(f"[Guardian] {symbol} критическая ошибка: {e}", exc_info=True)
        finally:
            logger.info(f"🛡️ [Guardian] Завершает активную фазу наблюдения за {symbol}.")

    async def _manage_averaging(self, symbol: str):
        cfg_dom = self.user_data.get("dom_squeeze_settings", config.DOM_SQUEEZE_STRATEGY)
        AVERAGING_STEP_PCT = 3.5
        MAX_AVERAGING_ORDERS = 1
        try:
            while symbol in self.open_positions:
                await asyncio.sleep(5)
                pos = self.open_positions.get(symbol)
                if not pos: return
                if self.averaging_orders_count.get(symbol, 0) >= MAX_AVERAGING_ORDERS:
                    logger.info(f"[{symbol}] Лимит усреднений ({MAX_AVERAGING_ORDERS}) достигнут. Управление усреднением завершено.")
                    return
                avg_price = self._resolve_avg_price(symbol, pos)
                last_price = utils.safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice"))
                if not (avg_price > 0 and last_price > 0): continue
                side = pos.get("side", "Buy")
                drawdown_pct = ((avg_price - last_price) / avg_price * 100.0) if side == "Buy" else ((last_price - avg_price) / avg_price * 100.0)
                if drawdown_pct >= AVERAGING_STEP_PCT:
                    logger.warning(f"🎯 [{symbol}] Просадка достигла {drawdown_pct:.2f}%. Инициирую усредняющий вход.")
                    candidate = {"symbol": symbol, "side": side, "source": "averaging_ladder", "is_averaging": True}
                    await self.execute_priority_trade(candidate)
                    return
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[AveragingManager] {symbol} критическая ошибка: {e}", exc_info=True)

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
            await self.place_unified_order(symbol=symbol, side=close_side, qty=qty, order_type="Market", comment=reason)
        except Exception as e:
            logger.error(f"[{symbol}] Ошибка при принудительном закрытии позиции: {e}", exc_info=True)

    async def secure_open_positions(
        self,
        reason: str = "Auto-secure pass",
        *,
        force_flatten: bool = False,
        sync_live: bool = False
    ):
        cfg = getattr(self, "position_security_cfg", config.POSITION_SECURITY)
        if not force_flatten and not cfg.get("ENABLED", True):
            return
        if sync_live:
            try:
                await self.update_open_positions()
            except Exception as exc:
                logger.warning(f"[PositionSecurity] Не удалось синхронизировать позиции перед '{reason}': {exc}")
        async with self.position_lock:
            snapshot = [(sym, dict(pos)) for sym, pos in self.open_positions.items()]
        if not snapshot:
            return

        threshold = utils.safe_to_float(cfg.get("PROFIT_THRESHOLD_PCT", 0.6))
        lock_ratio = utils.safe_to_float(cfg.get("LOCK_PERCENT_OF_PROFIT", 0.4))
        lock_cap = utils.safe_to_float(cfg.get("LOCK_CAP_PCT", 2.5))
        trail_buffer = utils.safe_to_float(cfg.get("TRAIL_BUFFER_PCT", 0.25))
        loss_cut = utils.safe_to_float(cfg.get("LOSS_HARD_STOP_PCT", 0.8))
        auto_close_losses = bool(cfg.get("AUTO_CLOSE_IF_LOSS", True))
        grace_sec = utils.safe_to_float(cfg.get("LOSS_GRACE_PERIOD_SEC", 0.0))
        protect_rating = utils.safe_to_float(cfg.get("LOSS_PROTECT_RATING", 0.0))
        protect_hold = utils.safe_to_float(cfg.get("LOSS_PROTECT_HOLD_RATIO", 0.0))

        tightened, closed = 0, 0
        now = time.time()

        for symbol, pos in snapshot:
            if pos.get("is_opening"):
                continue

            if pos.get("manual_mode"):
                logger.debug(f"[PositionSecurity] {symbol}: manual_mode активен, автоматическое сопровождение пропущено.")
                continue

            if force_flatten:
                await self.close_position(symbol, reason=f"{reason}: force flatten")
                closed += 1
                continue

            avg_price = self._resolve_avg_price(symbol, pos)
            side = str(pos.get("side", "")).lower()
            last_price = self.safe_last_price(symbol) or self.get_last_price(symbol)

            need_policy_features = bool(
                self.ml_policy
                and getattr(self.ml_policy, "enabled", False)
                and self.ml_policy.should_apply(pos.get("source", ""))
            )
            features_rt = None
            if last_price <= 0 or need_policy_features:
                try:
                    features_rt = await self.extract_realtime_features(symbol)
                except Exception:
                    features_rt = None
                if features_rt:
                    candidate_price = utils.safe_to_float(
                        features_rt.get("price")
                        or features_rt.get("lastPrice")
                        or features_rt.get("last_price")
                    )
                    if candidate_price > 0:
                        last_price = candidate_price
            if features_rt is None:
                features_rt = {}

            if avg_price <= 0 or side not in ("buy", "sell"):
                logger.warning(f"[PositionSecurity] {symbol}: пропуск, нет корректной информации о входе.")
                continue

            if last_price <= 0:
                logger.warning(f"[PositionSecurity] {symbol}: не удалось получить актуальную цену, стоп не обновлён.")
                continue

            if side == "buy":
                pnl_pct = ((last_price / avg_price) - 1.0) * 100.0
            else:
                pnl_pct = ((avg_price / last_price) - 1.0) * 100.0

            open_ts = utils.safe_to_float(pos.get("open_ts", 0.0))
            position_age = (now - open_ts) if open_ts > 0 else None
            wall_rating_meta = utils.safe_to_float(pos.get("wall_rating", 0.0))
            wall_hold_meta = utils.safe_to_float(pos.get("wall_hold_ratio", 0.0))
            if features_rt is not None:
                features_rt["live_pnl_pct"] = pnl_pct

            if self._policy_exit_decision(symbol, pos, features_rt):
                await self.close_position(symbol, reason=f"{reason}: ml-policy exit")
                closed += 1
                continue

            if pnl_pct >= threshold:
                lock_pct = max(threshold, min(lock_cap, pnl_pct * lock_ratio))
                if side == "buy":
                    candidate = avg_price * (1 + lock_pct / 100.0)
                    candidate = min(candidate, last_price * (1 - trail_buffer / 100.0))
                    candidate = max(candidate, avg_price * 1.0005)
                else:
                    candidate = avg_price * (1 - lock_pct / 100.0)
                    candidate = max(candidate, last_price * (1 + trail_buffer / 100.0))
                    candidate = min(candidate, avg_price * (1 - 0.0005))
                if candidate > 0:
                    try:
                        if await self.set_or_amend_stop_loss(candidate, symbol=symbol):
                            tightened += 1
                    except Exception as e:
                        logger.warning(f"[PositionSecurity] {symbol}: ошибка при подтяжке стопа ({e}).", exc_info=True)
            elif auto_close_losses and pnl_pct <= -loss_cut:
                if protect_rating > 0 and wall_rating_meta >= protect_rating:
                    logger.debug(f"[PositionSecurity] {symbol}: защита от закрытия — rating {wall_rating_meta} ≥ {protect_rating}.")
                    continue
                if protect_hold > 0 and wall_hold_meta >= protect_hold:
                    logger.debug(f"[PositionSecurity] {symbol}: защита от закрытия — hold {wall_hold_meta:.2f} ≥ {protect_hold:.2f}.")
                    continue
                if grace_sec > 0 and position_age is not None and position_age < grace_sec:
                    logger.debug(f"[PositionSecurity] {symbol}: защита от закрытия — возраст {position_age:.0f}s < {grace_sec}s.")
                    continue
                await self.close_position(symbol, reason=f"{reason}: limit loss {pnl_pct:.2f}%")
                closed += 1

        if tightened or closed:
            logger.info(f"[PositionSecurity] {reason}: tightened={tightened}, closed={closed}")

    async def _position_security_loop(self):
        cfg = getattr(self, "position_security_cfg", config.POSITION_SECURITY)
        interval = max(120, int(cfg.get("RECHECK_INTERVAL_SEC", 900)))
        try:
            while True:
                await asyncio.sleep(interval)
                await self.secure_open_positions(reason="Periodic auto-secure", sync_live=True)
        except asyncio.CancelledError:
            logger.debug("Position security loop cancelled.")
        except Exception as e:
            logger.error(f"Position security loop error: {e}", exc_info=True)
            try:
                await self.secure_open_positions(reason="Recovery auto-secure", sync_live=True)
            except Exception:
                logger.exception("Не удалось выполнить аварийную подтяжку стопов после ошибки цикла защиты.")


    async def save_dom_memory(self):
        """
        Сохраняет текущую dom_wall_memory в wall_memory.pkl
        """
        try:
            with open("wall_memory.pkl", "wb") as f:
                pickle.dump(self.dom_wall_memory, f)
            logger.debug("🧠 [Память Стен] Обновлена и сохранена в wall_memory.pkl.")
        except Exception as e:
            logger.error(f"[Память Стен] Ошибка при сохранении wall_memory.pkl: {e}", exc_info=True)


    async def _ai_advise_on_position(self, symbol: str):
        try:
            pos = self.open_positions.get(symbol)
            if not pos: return
            features = await self.extract_realtime_features(symbol)
            if not features: return
            avg_price = self._resolve_avg_price(symbol, pos)
            last_price = features.get("price", avg_price)
            if avg_price > 0:
                pnl = ((last_price / avg_price) - 1.0) if pos.get("side") == "Buy" else ((avg_price / last_price) - 1.0)
                pos['current_roi'] = pnl * 100.0 * (utils.safe_to_float(pos.get("leverage", 10.0)))
            prompt = ai_ml.build_position_management_prompt(symbol, pos, features)
            logger.info(f"🤖 [{symbol}] Запрос тактического совета у AI-Стратега...")
            ai_response = await ai_ml.ask_ollama_json(
                self.ai_advisor_model, [{"role": "user", "content": prompt}],
                timeout_s=45.0, base_url=self.ollama_advisor_openai
            )
            if ai_response.get("action") == "UPDATE_TACTICS":
                new_mult = utils.safe_to_float(ai_response.get("new_atr_multiplier"))
                new_tp = utils.safe_to_float(ai_response.get("take_profit_price"))
                reason = ai_response.get("reason", "N/A")
                side = pos.get("side")
                is_tp_valid = False
                if new_tp > 0 and last_price > 0:
                    deviation_pct = abs(new_tp - last_price) / last_price * 100
                    is_logical = (side == "Buy" and new_tp > avg_price) or (side == "Sell" and new_tp < avg_price)
                    is_sane = deviation_pct < 50.0
                    if is_logical and is_sane:
                        is_tp_valid = True
                    else:
                        logger.error(f"❌ [{symbol}] AI сгенерировал НЕВАЛИДНЫЙ TP! Цена: {new_tp:.6f}, Логично: {is_logical}, Адекватно: {is_sane}. Установка отменена.")
                logger.warning(f"🤖💡 [{symbol}] AI-Стратег обновил тактику: ATR x{new_mult}, TP={new_tp:.6f}. Причина: {reason}")
                if new_mult > 0:
                    pos["dynamic_atr_multiplier"] = new_mult
                if is_tp_valid:
                    await self.set_or_amend_stop_loss(0, symbol=symbol, take_profit_price=new_tp)
        except Exception as e:
            logger.error(f"[{symbol}] Ошибка в работе AI-Стратега: {e}", exc_info=True)

    async def _calculate_fibonacci_stop_price(self, symbol: str, side: str) -> Optional[float]:
        try:
            LOOKBACK_MINUTES = 180
            FIB_LEVEL = 0.618
            candles = list(self.shared_ws.candles_data.get(symbol, []))
            if len(candles) < LOOKBACK_MINUTES: return None
            recent_candles = candles[-LOOKBACK_MINUTES:]
            highest_high = max(utils.safe_to_float(c.get("highPrice")) for c in recent_candles)
            lowest_low = min(utils.safe_to_float(c.get("lowPrice")) for c in recent_candles)
            price_range = highest_high - lowest_low
            if price_range == 0: return None
            if side.lower() == "buy":
                return lowest_low + (price_range * FIB_LEVEL)
            else:
                return highest_high - (price_range * FIB_LEVEL)
        except Exception as e:
            logger.error(f"[{symbol}] Ошибка при расчете уровня Фибоначчи: {e}")
            return None

    async def _ai_advise_on_stop(self, symbol: str):
        try:
            pos = self.open_positions.get(symbol)
            if not pos: return
            features = await self.extract_realtime_features(symbol)
            if not features:
                logger.warning(f"[{symbol}] Не удалось получить фичи для AI-советника.")
                return
            pos['last_stop_price'] = self.last_stop_price.get(symbol)
            prompt = ai_ml.build_position_management_prompt(symbol, pos, features)
            messages = [{"role": "user", "content": prompt}]
            logger.info(f"🤖 [{symbol}] Запрос совета у AI-риск-менеджера...")
            ai_response = await ai_ml.ask_ollama_json(
                self.ai_advisor_model, messages, timeout_s=45.0, base_url=self.ollama_advisor_openai
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

    async def _set_initial_stop_loss(self, symbol: str, pos: dict, force: bool = False):
        """
        [V7] Добавлена проверка на race condition:
        стоп не будет установлен, если цена уже ушла за его пределы.
        """
        try:
            if pos.get("initial_stop_set") and not force:
                return

            avg_price = self._resolve_avg_price(symbol, pos)
            if avg_price <= 0:
                logger.error(f"[{symbol}] Не удалось получить цену входа для установки стопа.")
                return

            side = str(pos.get("side", "")).lower()
            features = await self.extract_realtime_features(symbol)
            if not features or not features.get("atr14"):
                logger.warning(f"[{symbol}] Не удалось получить ATR. Используется fallback.")
                atr_val = avg_price * 0.05
            else:
                atr_val = float(features.get("atr14", 0.0))

            atr_multiplier = self.user_data.get("INITIAL_STOP_ATR_MULTIPLIER", 3.0)
            stop_distance = atr_val * atr_multiplier
            stop_price_by_atr = avg_price - stop_distance if side == "buy" else avg_price + stop_distance

            leverage = utils.safe_to_float(
                pos.get("leverage")
                or self.user_data.get("leverage")
                or getattr(self, "leverage", 1.0)
            )
            leverage = max(leverage, 1.0)

            max_stop_pct = self.user_data.get("MAX_INITIAL_STOP_PCT", 4.0) / 100.0
            max_levered_loss_pct = utils.safe_to_float(
                self.user_data.get("MAX_LEVERED_LOSS_PCT"),
                getattr(config, "MAX_LEVERED_LOSS_PCT", 30.0),
            )
            if max_levered_loss_pct > 0:
                risk_price_pct = (max_levered_loss_pct / leverage) / 100.0
                max_stop_pct = min(max_stop_pct, risk_price_pct)

            max_stop_price = avg_price * (1 - max_stop_pct) if side == "buy" else avg_price * (1 + max_stop_pct)

            if side == "buy":
                final_stop_price = max(stop_price_by_atr, max_stop_price)
            else:
                final_stop_price = min(stop_price_by_atr, max_stop_price)

            last_price = self.safe_last_price(symbol)
            stop_to_apply = final_stop_price

            if max_levered_loss_pct > 0:
                risk_price_pct = (max_levered_loss_pct / leverage) / 100.0
                if side == "buy":
                    risk_stop = avg_price * (1 - risk_price_pct)
                    stop_to_apply = max(stop_to_apply, risk_stop)
                else:
                    risk_stop = avg_price * (1 + risk_price_pct)
                    stop_to_apply = min(stop_to_apply, risk_stop)

            min_stop_pct = utils.safe_to_float(
                self.user_data.get("MIN_INITIAL_STOP_PCT"),
                getattr(config, "MIN_INITIAL_STOP_PCT", 0.0)
            ) / 100.0
            if min_stop_pct > 0:
                if side == "buy":
                    min_stop_price = avg_price * (1 - min_stop_pct)
                    stop_to_apply = min(stop_to_apply, min_stop_price)
                else:
                    min_stop_price = avg_price * (1 + min_stop_pct)
                    stop_to_apply = max(stop_to_apply, min_stop_price)

            def _safety_stop() -> float | None:
                safety_pct = utils.safe_to_float(
                    self.user_data.get("max_safety_stop_pct")
                    or self.user_data.get("MAX_SAFETY_STOP_PCT", 15.0), 15.0
                ) / 100.0
                # доп. буфер чтобы стоп был ниже/выше текущей цены на пару тиков
                fallback_buffer_pct = utils.safe_to_float(
                    self.user_data.get("INITIAL_STOP_FALLBACK_BUFFER_PCT", 0.25), 0.25
                ) / 100.0

                if side == "buy":
                    safety_price = avg_price * (1 - safety_pct)
                    upper_bound = last_price * (1 - fallback_buffer_pct) if last_price > 0 else safety_price
                    if upper_bound <= 0:
                        return None
                    if safety_price <= 0:
                        return upper_bound
                    if upper_bound < safety_price:
                        return None  # рынок уже глубже лимита безопасности
                    return upper_bound
                else:
                    safety_price = avg_price * (1 + safety_pct)
                    lower_bound = last_price * (1 + fallback_buffer_pct) if last_price > 0 else safety_price
                    if lower_bound <= 0:
                        return None
                    if lower_bound > safety_price:
                        return None  # цена уже ушла выше допустимого уровня риска
                    return lower_bound

            if last_price > 0:
                crossed = (side == "buy" and final_stop_price >= last_price) or \
                          (side == "sell" and final_stop_price <= last_price)
                if crossed:
                    logger.warning(
                        f"[{symbol}] Рынок перешагнул расчётный SL "
                        f"(calc={final_stop_price:.6f}, last={last_price:.6f}). Пытаюсь применить безопасный стоп."
                    )
                    safety_stop = _safety_stop()
                    if safety_stop:
                        stop_to_apply = safety_stop
                    else:
                        logger.error(f"[{symbol}] Не удалось вычислить безопасный стоп. Принудительно закрываю позицию.")
                        await self.close_position(symbol, reason="Initial stop unavailable; forced exit")
                        pos["initial_stop_set"] = False
                        return

            if (not stop_to_apply or stop_to_apply <= 0 or
                (side == "buy" and last_price > 0 and stop_to_apply >= last_price) or
                (side == "sell" and last_price > 0 and stop_to_apply <= last_price) or
                (side == "buy" and stop_to_apply >= avg_price) or
                (side == "sell" and stop_to_apply <= avg_price)):
                logger.error(f"[{symbol}] КРИТИЧЕСКАЯ ОШИБКА РАСЧЁТА СТОПА. Установка отменена.")
                return

            logger.info(f"🛡️ [{symbol}] Установка первоначального SL. Entry: {avg_price:.6f}, SL: {stop_to_apply:.6f}")
            success = await self.set_or_amend_stop_loss(stop_to_apply, symbol=symbol)
            pos["initial_stop_set"] = bool(success)
            if not success:
                logger.error(f"[{symbol}] API не подтвердил установку начального стопа. Позиция остаётся без защиты.")
        except Exception as e:
            logger.error(f"[{symbol}] Ошибка _set_initial_stop_loss: {e}", exc_info=True)




    async def set_or_amend_stop_loss(
        self,
        price: float,
        *,
        symbol: str,
        cancel_only: bool = False,
        take_profit_price: Optional[float] = None
    ):
        """
        [ФИНАЛЬНАЯ ВЕРСИЯ v2] Устанавливает/изменяет SL/TP с корректным positionIdx.
        """
        self._ensure_trailing_state(symbol)
        now = time.time()
        pos = self.open_positions.get(symbol, {})
        side = (pos.get("side", "Buy") or "Buy").lower()
        previous_stop = utils.safe_to_float(self.last_sent_stop_price.get(symbol))
        last_attempt = self.last_stop_attempt_ts.get(symbol, 0.0)

        if not cancel_only and now - last_attempt < self.min_sl_retry_sec:
            return False

        payload = {"category": "linear", "symbol": symbol}
        
        # Правильно определяем positionIdx для One-Way и Hedge Mode
        if self.position_mode == 0: # One-Way Mode
            payload["positionIdx"] = 0
        else: # Hedge Mode
            payload["positionIdx"] = 1 if side == "buy" else 2

        if cancel_only:
            payload["stopLoss"] = "0"
        elif price > 0:
            stop_price = self._round_to_tick(symbol, float(price), side)
            last_sent = float(self.last_sent_stop_price.get(symbol, 0.0) or 0.0)
            if abs(stop_price - last_sent) > 1e-9:
                payload["stopLoss"] = f"{stop_price:.10f}".rstrip("0").rstrip(".")

        if take_profit_price and take_profit_price > 0:
            tp_price = self._round_to_tick(symbol, float(take_profit_price), "sell" if side == "buy" else "buy")
            payload["takeProfit"] = f"{tp_price:.10f}".rstrip("0").rstrip(".")

        if "stopLoss" in payload and not cancel_only:
            raw_stop = utils.safe_to_float(payload["stopLoss"])
            tick = self._get_tick(symbol)
            tick = tick if tick > 0 else 1e-6
            avg_price = utils.safe_to_float(pos.get("avg_price") or pos.get("avgPrice") or 0.0)
            last_price = self.safe_last_price(symbol)

            adjusted_stop = raw_stop
            changed = False

            trailing_active = bool(pos.get("trailing_activated")) or bool(self.trailing_activated.get(symbol))

            if side == "sell":
                min_allowed = None
                if last_price and last_price > 0:
                    min_allowed = last_price + tick
                if not trailing_active and avg_price and avg_price > 0:
                    base_limit = avg_price + tick
                    min_allowed = max(min_allowed or 0.0, base_limit)
                if min_allowed and adjusted_stop <= min_allowed:
                    adjusted_stop = min_allowed
                    changed = True
            else:  # buy
                if last_price and last_price > 0:
                    max_allowed = last_price - tick
                    if adjusted_stop >= max_allowed:
                        adjusted_stop = max_allowed
                        changed = True

            if adjusted_stop <= 0:
                logger.warning(f"[{symbol}] stopLoss calc <=0 ({adjusted_stop}). Команда SL пропущена.")
                payload.pop("stopLoss", None)
            else:
                adjusted_stop = self._round_to_tick(symbol, adjusted_stop, side)
                if changed and abs(adjusted_stop - raw_stop) > 1e-9:
                    logger.debug(
                        f"[SL_GUARD] {symbol} {side.upper()} stop скорректирован {raw_stop:.10f} -> {adjusted_stop:.10f}"
                        f" (avg={avg_price:.10f}, last={last_price:.10f})"
                    )
                    payload["stopLoss"] = f"{adjusted_stop:.10f}".rstrip("0").rstrip(".")

        if "stopLoss" not in payload and "takeProfit" not in payload:
            return False

        logger.info(f"⚙️ [{symbol}] Отправка команды SL/TP: {payload}")
        if not cancel_only:
            self.last_stop_attempt_ts[symbol] = now
        success = False
        if "stopLoss" in payload:
            self.last_sent_stop_price[symbol] = utils.safe_to_float(payload["stopLoss"])
        try:
            resp = await asyncio.to_thread(self.session.set_trading_stop, **payload)
            
            if resp.get("retCode") == 0:
                logger.info(f"✅ [{symbol}] API подтвердил установку SL/TP.")
                if "stopLoss" in payload:
                    sent_sl = utils.safe_to_float(payload["stopLoss"])
                    self.last_sent_stop_price[symbol] = sent_sl
                    self.last_stop_price[symbol] = sent_sl
                    if pos is not None:
                        pos["last_stop_price"] = sent_sl
                success = True
            else:
                self.last_sent_stop_price[symbol] = self.last_stop_price.get(symbol, 0.0)
                logger.warning(f"[{symbol}] API вернул ошибку при установке SL/TP: {resp.get('retMsg')}")
                success = False

        except InvalidRequestError as e:
            msg = str(e).lower()
            if "not modified" in msg or "34040" in msg:
                logger.info(f"[{symbol}] SL/TP уже установлен на этом уровне.")
                if "stopLoss" in payload:
                    self.last_stop_price[symbol] = utils.safe_to_float(payload["stopLoss"])
                    success = True
            else:
                self.last_sent_stop_price[symbol] = self.last_stop_price.get(symbol, 0.0)
                logger.warning(f"[{symbol}] Ошибка API InvalidRequestError: {e}")
                success = False
        except Exception as e:
            self.last_sent_stop_price[symbol] = self.last_stop_price.get(symbol, 0.0)
            logger.error(f"[{symbol}] Критическая ошибка в set_or_amend_stop_loss: {e}", exc_info=True)
            success = False
        finally:
            if not success and previous_stop > 0 and not cancel_only:
                # Возвращаем предыдущее значение в локальный кэш, чтобы повторить позже
                self.last_sent_stop_price[symbol] = previous_stop
                logger.warning(f"[{symbol}] Возвращаю предыдущее значение SL ({previous_stop}) из-за неуспеха обновления.")

        return success


    async def _run_adaptive_trailing(self, symbol: str, last_price: float, pos: Dict[str, Any]):
        """
        Адаптивный трейлинг на основе фаз движения рынка (AGGR_START / SLOW_TREND / RANGE),
        ROI и профиля инструмента из config.TRAILING_PROFILES.
        """
        try:
            self._ensure_trailing_state(symbol)

            profile = self._get_trailing_profile(symbol)
            atr_len = int(profile.get("ATR_LEN", getattr(config, "ATR_LEN", 14)))
            tb_min = float(profile.get("TB_MIN", getattr(config, "TB_MIN", 0.7)))
            tb_max = float(profile.get("TB_MAX", getattr(config, "TB_MAX", 3.2)))
            roi_activation = float(profile.get("ROI_ACTIVATION", getattr(config, "TRAILING_ACTIVATION_ROI", 0.05)))
            cooldown_sec = float(profile.get("COOLDOWN_SEC", getattr(config, "TRAILING_COOLDOWN_SEC", 1.0)))
            min_update_interval = float(profile.get("MIN_UPDATE_INTERVAL_SEC", cooldown_sec))
            spiky_tail_ratio = float(profile.get("SPIKY_TAIL_RATIO", 0.66))
            atr_fallback_pct = float(profile.get("ATR_FALLBACK_PCT", 0.015))

            now = time.time()
            if now - self.last_trailing_update_ts.get(symbol, 0.0) < min_update_interval:
                return

            lock = self.trailing_lock[symbol]
            if lock.locked():
                return

            async with lock:
                pos = self.open_positions.get(symbol) or {}
                avg_price = self._resolve_avg_price(symbol, pos)
                if avg_price <= 0 or last_price <= 0:
                    return

                side = str(pos.get("side", "Buy")).lower()
                if side not in ("buy", "sell"):
                    return

                peak = pos.get("trailing_peak")
                peak_ts = pos.get("trailing_peak_ts", now)
                if peak is None:
                    peak = avg_price
                    peak_ts = now
                if side == "buy":
                    if last_price > peak:
                        peak = last_price
                        peak_ts = now
                else:
                    if last_price < peak:
                        peak = last_price
                        peak_ts = now
                pos["trailing_peak"] = peak
                pos["trailing_peak_ts"] = peak_ts

                df = pos.get("ohlcv")
                features_rt: dict[str, Any] | None = None
                min_required = max(atr_len + 5, 20)
                lookback = max(atr_len * 6, 120)
                use_df = False
                atr_val: float | None = None
                confidence = 0.0
                metrics: Dict[str, Any] = {}
                regime = "SLOW_TREND"

                if df is not None and len(df) >= min_required:
                    df = df.tail(lookback).reset_index(drop=True)
                    use_df = True
                else:
                    df = self._get_recent_ohlcv_1m(symbol, lookback=lookback)
                    if df is not None and len(df) >= min_required:
                        use_df = True

                if use_df:
                    atr_series = ta.atr(df["high"], df["low"], df["close"], length=atr_len)
                    if atr_series is not None and not atr_series.empty:
                        atr_val = float(atr_series.iloc[-1])
                    if atr_val is None or not np.isfinite(atr_val) or atr_val <= 0:
                        use_df = False
                        atr_val = None

                if use_df:
                    closes = df["close"].to_numpy(dtype=float)
                    volumes = df["volume"].to_numpy(dtype=float)
                    if len(closes) >= 20:
                        closes_slice = closes[-60:] if len(closes) >= 60 else closes
                        volumes_slice = volumes[-len(closes_slice):]
                        regime, confidence, metrics = detect_market_regime(closes_slice, volumes_slice, atr_val)
                    else:
                        use_df = False
                        atr_val = None
                        metrics = {}
                        regime = "SLOW_TREND"

                if not use_df:
                    features_rt = await self.extract_realtime_features(symbol)
                    if features_rt:
                        atr_val = float(features_rt.get("atr14") or 0.0)
                    if atr_val is None or not np.isfinite(atr_val) or atr_val <= 0:
                        atr_val = max(last_price * atr_fallback_pct, 1e-6)
                    regime = "SLOW_TREND"
                    metrics = {"slope": 0.0, "dsr": 0.0, "nsr": 2.0, "vol_boost": 1.0, "vol_ratio": 1.0, "source": "fallback"}

                if features_rt is None:
                    try:
                        features_rt = await self.extract_realtime_features(symbol)
                    except Exception:
                        features_rt = {}
                else:
                    features_rt = dict(features_rt)

                if side == "buy":
                    roi_unlev = (last_price - avg_price) / max(avg_price, 1e-12)
                else:
                    roi_unlev = (avg_price - last_price) / max(avg_price, 1e-12)

                lev = utils.safe_to_float(
                    pos.get("leverage")
                    or getattr(self, "symbol_leverage_map", {}).get(symbol)
                    or profile.get("ASSUME_LEVERAGE_IF_MISSING")
                    or config.TRAILING_MODES.get("dynamic", {}).get("ASSUME_LEVERAGE_IF_MISSING")
                    or 1.0
                )
                lev = max(1.0, lev)
                roi_lev = roi_unlev * lev

                pos["roi"] = roi_unlev
                pos["roi_lev"] = roi_lev

                start_roi_pct, gap_roi_pct = self._get_trailing_params(symbol)
                start_roi_unlev = max((start_roi_pct / 100.0) / lev, 0.0)
                gap_roi_unlev = max((gap_roi_pct / 100.0) / lev, 0.0)
                roi_activation = max(roi_activation, start_roi_unlev)

                k_map = profile.get("K", {})
                k_min, k_max = k_map.get(regime, k_map.get("SLOW_TREND", (1.5, 2.3)))
                tb = float(atr_val) * ((float(k_min) + float(k_max)) / 2.0)

                if roi_unlev > roi_activation:
                    tb *= max(0.65, 1.0 - (roi_unlev - roi_activation) * 2.0)
                elif roi_unlev < 0:
                    tb *= 1.10

                slope = float(metrics.get("slope", 0.0))
                dsr = float(metrics.get("dsr", 0.0))
                nsr = float(metrics.get("nsr", 2.0))

                dir_score = 0.0
                dir_score += np.clip((slope - 0.25) / 0.25, 0.0, 1.0) * 0.5
                dir_score += np.clip((dsr - 0.60) / 0.20, 0.0, 1.0) * 0.3
                noise_pen = np.clip((nsr - 1.7) / 0.8, 0.0, 1.0) * 0.3
                tb *= (1.0 - dir_score * 0.35 + noise_pen * 0.35)

                spike_tighten_mult = float(profile.get("SPIKE_TIGHTEN_MULT", getattr(config, "TRAILING_SPIKE_TIGHTEN_MULT", 0.55)))
                spike_force_ticks = float(profile.get("SPIKE_FORCE_TICKS", getattr(config, "TRAILING_SPIKE_FORCE_TICKS", 2)))
                spike_vol_threshold = float(profile.get("SPIKE_VOL_THRESHOLD", getattr(config, "TRAILING_SPIKE_VOL_THRESHOLD", 4.0)))
                spike_price_threshold = float(profile.get("SPIKE_PRICE_THRESHOLD", getattr(config, "TRAILING_SPIKE_PRICE_THRESHOLD", 1.0)))

                vol_anom = float(utils.safe_to_float((features_rt or {}).get("volume_anomaly"), 1.0))
                pct1 = utils.safe_to_float((features_rt or {}).get("pct1m"), 0.0)
                pct5 = utils.safe_to_float((features_rt or {}).get("pct5m"), 0.0)

                bar_move_pct = 0.0
                bar_dir_pct = 0.0
                if use_df and df is not None and len(df) > 0:
                    last_bar = df.iloc[-1]
                    bar_open = float(last_bar["open"])
                    bar_close = float(last_bar["close"])
                    bar_high = float(last_bar["high"])
                    bar_low = float(last_bar["low"])
                    base_price = max(avg_price, 1e-9)
                    bar_move_pct = abs(bar_close - bar_open) / base_price * 100.0
                    bar_range_pct = abs(bar_high - bar_low) / base_price * 100.0
                    bar_dir_pct = (bar_close - bar_open) / base_price * 100.0
                else:
                    bar_range_pct = 0.0

                if side == "buy":
                    favorable_pct = max(pct1, pct5, bar_dir_pct, 0.0)
                else:
                    favorable_pct = max(-pct1, -pct5, -bar_dir_pct, 0.0)

                extreme_spike = False
                if vol_anom >= spike_vol_threshold:
                    extreme_spike = True
                if favorable_pct >= spike_price_threshold:
                    extreme_spike = True
                if bar_range_pct >= spike_price_threshold * 1.5:
                    extreme_spike = True

                if extreme_spike:
                    shrink = max(0.15, min(spike_tighten_mult, 1.0))
                    tb *= shrink
                    tb = max(tb, tb_min * atr_val * 0.35)

                tb = float(np.clip(tb, tb_min * atr_val, tb_max * atr_val))
                force_update = bool(extreme_spike)

                last_extreme = float(peak)
                new_stop = (last_extreme - tb) if side == "buy" else (last_extreme + tb)

                tick = self._get_tick(symbol) or 1e-6
                be_buffer_pct = float(profile.get("BREAKEVEN_BUFFER_PCT", 0.18)) / 100.0
                be_buffer_pct = max(be_buffer_pct, gap_roi_unlev)
                trailing_active = self.trailing_activated.get(symbol, False)
                if trailing_active:
                    if side == "buy":
                        breakeven_price = avg_price * (1.0 + be_buffer_pct)
                        fail_fast_be = utils.safe_to_float(pos.get("fail_fast_be_price"), 0.0)
                        if fail_fast_be > 0:
                            breakeven_price = max(breakeven_price, fail_fast_be)
                        if last_price <= breakeven_price + tick:
                            return
                        if (last_price - tick) <= breakeven_price:
                            return
                        new_stop = max(new_stop, breakeven_price)
                    else:
                        breakeven_price = avg_price * (1.0 - be_buffer_pct)
                        fail_fast_be = utils.safe_to_float(pos.get("fail_fast_be_price"), 0.0)
                        if fail_fast_be > 0:
                            breakeven_price = min(breakeven_price, fail_fast_be)
                        if last_price >= breakeven_price - tick:
                            return
                        if (last_price + tick) >= breakeven_price:
                            return
                        new_stop = min(new_stop, breakeven_price)

                profit_unlev = (peak - avg_price) / max(avg_price, 1e-9) if side == "buy" else (avg_price - peak) / max(avg_price, 1e-9)
                if gap_roi_unlev > 0 and profit_unlev >= gap_roi_unlev:
                    if side == "buy":
                        trailing_gap_stop = peak * (1.0 - gap_roi_unlev)
                        new_stop = max(new_stop, trailing_gap_stop)
                    else:
                        trailing_gap_stop = peak * (1.0 + gap_roi_unlev)
                        new_stop = min(new_stop, trailing_gap_stop)

                if side == "buy":
                    new_stop = min(new_stop, last_price - tick)
                    new_stop = math.floor(new_stop / tick) * tick
                else:
                    new_stop = max(new_stop, last_price + tick)
                    new_stop = math.ceil(new_stop / tick) * tick

                if side == "buy" and new_stop <= avg_price:
                    return
                if side == "sell" and new_stop >= avg_price:
                    return

                current_stop = utils.safe_to_float(self.last_stop_price.get(symbol, 0.0))
                if current_stop > 0:
                    if side == "buy" and new_stop <= current_stop + tick * 0.5:
                        return
                    if side == "sell" and new_stop >= current_stop - tick * 0.5:
                        return

                moved_enough = abs(new_stop - current_stop) >= tick if current_stop > 0 else True
                if not moved_enough:
                    return

                if not self.trailing_activated.get(symbol, False) and roi_unlev >= start_roi_unlev:
                    self.trailing_activated[symbol] = True
                    self.trailing_activation_ts[symbol] = now

                if not self.trailing_activated.get(symbol, False):
                    return

                if use_df and df is not None:
                    pos["ohlcv"] = df

                if use_df and len(df) > 0:
                    last_bar = df.iloc[-1]
                    body = abs(float(last_bar["close"]) - float(last_bar["open"]))
                    full_range = float(last_bar["high"]) - float(last_bar["low"])
                    tail_ratio = 0.0 if full_range <= 0 else (full_range - body) / full_range
                    spiky = tail_ratio > spiky_tail_ratio
                else:
                    spiky = False

                last_log_ts = float(self._trailing_log_ts.get(symbol, 0.0))
                cooldown_hit = (now - last_log_ts) < cooldown_sec
                if not force_update and spiky and current_stop > 0:
                    delta_ticks = abs(new_stop - current_stop) / tick
                    if delta_ticks >= spike_force_ticks:
                        force_update = True
                if not force_update and cooldown_hit:
                    return
                if not force_update and spiky:
                    return
                if force_update:
                    logger.debug(f"[TRAIL_SPIKE] {symbol}: форсируем обновление SL (delta={abs(new_stop-current_stop):.6f}, vol_anom={vol_anom:.2f}, move_pct={favorable_pct:.2f})")

                prev_stop_logged = utils.safe_to_float(self.last_stop_price.get(symbol, 0.0))
                if prev_stop_logged > 0 and abs(new_stop - prev_stop_logged) < tick * 0.5:
                    return

                success = await self.set_or_amend_stop_loss(new_stop, symbol=symbol)
                if success:
                    self.last_trailing_update_ts[symbol] = now
                    self._trailing_log_ts[symbol] = now
                    pos["last_adaptive_update"] = {
                        "roi": roi_unlev,
                        "roi_lev": roi_lev,
                        "regime": regime,
                        "confidence": confidence,
                        "tb": tb,
                        "atr": atr_val,
                        "extreme": last_extreme,
                        "source": "ohlcv" if use_df else "fallback",
                        "vol_anom": vol_anom,
                        "move_pct": favorable_pct,
                        "forced": force_update,
                    }
                    logger.info(
                        f"[{symbol}] Adaptive trailing: ROI_price={roi_unlev:.2%} "
                        f"(ROI_lev={roi_lev:.2%}), {summarize_regime(regime, metrics)}"
                    )
                    try:
                        with open("trend_log.csv", "a") as f:
                            f.write(
                                f"{symbol},{roi_lev:.6f},{regime},"
                                f"{metrics.get('slope',0):.6f},{metrics.get('dsr',0):.6f},"
                                f"{metrics.get('nsr',0):.6f},{metrics.get('vol_boost',0):.6f},"
                                f"{metrics.get('vol_ratio',0):.6f},{now}\n"
                            )
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"Adaptive trailing error for {symbol}: {e}", exc_info=True)





    def _purge_symbol_state(self, symbol: str):
        logger.debug(f"Полная очистка состояния для символа: {symbol}")
        if task := self.watch_tasks.pop(symbol, None):
            if not task.done():
                task.cancel()
                logger.debug(f"[{symbol}] Guardian task отменен.")
        pos = self.open_positions.pop(symbol, None)
        if isinstance(pos, dict):
            pos.pop("trailing_peak_ts", None)
        self.last_stop_price.pop(symbol, None)
        self.pending_orders.pop(symbol, None)
        self.recently_closed[symbol] = time.time()
        self.trailing_activated.pop(symbol, None)
        self.trailing_activation_ts.pop(symbol, None)
        self.take_profit_price.pop(symbol, None)
        self.position_peak_price.pop(symbol, None)
        self.open_positions.pop(symbol, {}).pop("trailing_peak_ts", None)
        self.averaging_orders_count.pop(symbol, None)
        self.pending_signal_ids.pop(symbol, None)
        self.pending_position_meta.pop(symbol, None)
        self.pending_scale_ins.pop(symbol, None)
        self.pending_policy_vectors.pop(symbol, None)
        self.pending_policy_records.pop(symbol, None)
        self.learning_positions.pop(symbol, None)
        task = self._fail_fast_tasks.pop(symbol, None)
        if task and not task.done():
            task.cancel()
        self._snapshot_open_positions()

    async def _cleanup_pnl_cache(self, interval: int = 60, max_age: int = 300):
        while True:
            await asyncio.sleep(interval)
            now = time.time()
            expired_symbols = [s for s, data in self.recently_closed_pnl_cache.items() if now - data.get("timestamp", 0) > max_age]
            for symbol in expired_symbols:
                self.recently_closed_pnl_cache.pop(symbol, None)

    async def _cleanup_recently_closed(self, interval: int = 15, max_age: int = 60):
        while True:
            await asyncio.sleep(interval)
            now = time.time()
            expired = [s for s, ts in self.recently_closed.items() if now - ts > max_age]
            for s in expired:
                self.recently_closed.pop(s, None)



    async def on_ticker_update(
        self,
        symbol: str,
        last_price: float,
        best_bid: float | None = None,
        best_ask: float | None = None,
        tick: dict | None = None,
    ):
        # Обновляем карты last/bid/ask
        try:
            if last_price and last_price > 0:
                self.last_price_map[symbol] = last_price

            if best_bid is not None and best_bid > 0:
                self.best_bid_map[symbol] = best_bid
            elif tick:
                bb = utils.safe_to_float(tick.get("bid1Price") or tick.get("bestBidPrice") or tick.get("bidPrice") or 0.0)
                if bb > 0:
                    self.best_bid_map[symbol] = bb

            if best_ask is not None and best_ask > 0:
                self.best_ask_map[symbol] = best_ask
            elif tick:
                ba = utils.safe_to_float(tick.get("ask1Price") or tick.get("bestAskPrice") or tick.get("askPrice") or 0.0)
                if ba > 0:
                    self.best_ask_map[symbol] = ba
        except Exception:
            logger.exception(f"[{symbol}] Не удалось обновить карты bid/ask из тикера.")
        
        pos = self.open_positions.get(symbol)
        
        # Если есть открытая позиция, управляем ее трейлингом
        if pos and pos.get("initial_stop_set") and not pos.get("is_closing"):
            await self._run_trailing_stop_logic(symbol, last_price, pos)

        # Golden Setup – запускаем по живым данным с троттлингом
        try:
            if self.strategy_mode in ("full", "golden_only", "golden_squeeze"):
                now_ts = time.time()
                last_seen = self._golden_tick_ts.get(symbol, 0.0)
                if now_ts - last_seen >= self.golden_tick_interval_sec:
                    self._golden_tick_ts[symbol] = now_ts
                    if self._golden_allowed(symbol):
                        asyncio.create_task(strategies.golden_strategy(self, symbol))
        except Exception:
            logger.exception(f"[{symbol}] Ошибка запуска golden_strategy из тикового обработчика.")

        try:
            if self.aggressive_golden_enabled and self.strategy_mode in ("full", "golden_only", "golden_squeeze"):
                now_ts = time.time()
                last_tick = self._aggressive_tick_ts.get(symbol, 0.0)
                if now_ts - last_tick >= getattr(self, "aggressive_golden_tick_interval", 2.0):
                    self._aggressive_tick_ts[symbol] = now_ts
                    asyncio.create_task(strategies.aggressive_golden_setup(self, symbol))
        except Exception:
            logger.exception(f"[{symbol}] Ошибка запуска aggressive_golden_setup из тикового обработчика.")

        # === МЕЖСЕССИОННЫЕ СТРАТЕГИИ (21:00..00:00 UTC) ===
        if self.intersession_enabled:
            try:
                # работаем ТОЛЬКО если позиции НЕТ и нет pending
                if pos:
                    return
                if symbol in self.pending_orders or symbol in self.pending_signal_ids:
                    return
                check_time = dt.datetime.utcnow()
                if not is_quiet_market_window(check_time):
                    return

                throttle_cfg = getattr(config, "INTERSESSION_CONFIG", {}).get("THROTTLE", {})
                throttle = float(throttle_cfg.get("BASE_SEC", getattr(config, "INTERSESSION_THROTTLE_SEC", 15.0)))
                now = time.time()
                last_check = float(self._intersession_last_check_ts.get(symbol, 0.0))
                if now - last_check < throttle:
                    return
                self._intersession_last_check_ts[symbol] = now

                # Берем 1m OHLCV из WS
                df = self._get_recent_ohlcv_1m(symbol, lookback=60)
                if df is None or len(df) < 30:
                    return

                spread_pct = float(self._orderbook_metrics.get(symbol, {}).get("orderbook_implied_spread_pct", 0.0) or 0.0)
                orderbook_metrics = dict(self._orderbook_metrics.get(symbol, {}))
                features = await self.extract_realtime_features(symbol) or {}
                signals = generate_intersession_signals(
                    symbol,
                    df,
                    now=check_time,
                    market_snapshot={
                        "spread_pct": spread_pct,
                        "orderbook": orderbook_metrics,
                        "features": features,
                    },
                )
                if not signals:
                    return

                for s in signals:
                    kind = s.get("source","")
                    side = s.get("side","")
                    if side in ("Buy","Sell"):
                        candidate = {
                            "symbol": symbol,
                            "side": side,
                            "source": kind,
                            "justification": s.get("justification",""),
                            "trailing_mode": "adaptive",   # наше управление поз.
                        }
                        features = {"ticker": {"lastPrice": last_price}}
                        await self.execute_trade_entry(candidate, features)
                        logger.info(f"[{symbol}] INTRAQUIET entry placed: {kind} {side}")
                        self._intersession_last_signal_ts[symbol] = now
                        break
                    elif side == "GRID" and "grid_levels" in s:
                        buys, sells = s["grid_levels"]
                        await self._place_grid_orders(symbol, buys, sells, last_price)
                        self._intersession_last_signal_ts[symbol] = now
                        break

            except Exception as e:
                logger.error(f"[{symbol}] intersession strategies error: {e}", exc_info=True)



    async def _run_trailing_stop_logic(self, symbol: str, last_price: float, pos: dict):
        """
        Управляет трейлингом в зависимости от выбранного
        в config.py режима: 'dynamic' / 'simple_gap' / 'adaptive'.
        """
        pos_mode = (pos or {}).get("trailing_mode")
        user_mode = self.user_data.get("trailing_mode")
        mode = str(pos_mode or user_mode or config.ACTIVE_TRAILING_MODE or "simple_gap").lower()

        if mode == "simple_gap":
            await self._run_simple_gap_trailing(symbol, last_price, pos)
        elif mode == "dynamic":
            await self._run_dynamic_atr_trailing(symbol, last_price, pos)
        elif mode == "adaptive":
            await self._run_adaptive_trailing(symbol, last_price, pos)
        else:
            # Фолбэк: ничего не делаем
            return


    async def _run_simple_gap_trailing(self, symbol: str, last_price: float, pos: dict):
        """
        Реализует трейлинг с гарантированным процентным отступом.
        """
        self._ensure_trailing_state(symbol)
        tcfg = config.TRAILING_MODES.get("simple_gap", {})
        log_throttle = float(tcfg.get("LOG_THROTTLE_SEC", 2.0))
        min_log_delta = max(1, int(tcfg.get("MIN_LOG_DELTA_TICKS", 1)))
        
        now = time.time()
        if now - self.last_trailing_update_ts.get(symbol, 0) < 0.3:
            return

        lock = self.trailing_lock[symbol]
        if lock.locked(): return

        async with lock:
            pos = self.open_positions.get(symbol)
            if not pos: return

            avg_price = self._resolve_avg_price(symbol, pos)
            if avg_price <= 0: return

            side = (pos.get("side", "Buy") or "Buy").lower()
            
            pnl_pct = (((last_price - avg_price) / avg_price) * 100) if side == "buy" else (((avg_price - last_price) / avg_price) * 100)

            if not pos.get("trailing_activated") and pnl_pct >= tcfg.get("ACTIVATION_PNL_PCT", 1.0):
                pos["trailing_activated"] = True
                logger.info(f"✅ [{symbol}] Простой трейлинг АКТИВИРОВАН. PnL достиг {pnl_pct:.2f}%.")

            if not pos.get("trailing_activated"):
                return

            if "trailing_peak" not in pos: pos["trailing_peak"] = avg_price
            
            if (side == "buy" and last_price > pos["trailing_peak"]) or \
               (side == "sell" and last_price < pos["trailing_peak"]):
                pos["trailing_peak"] = last_price
            
            peak_price = pos["trailing_peak"]
            
            gap_pct = tcfg.get("TRAILING_GAP_PCT", 0.5)
            if side == "buy":
                candidate_stop = peak_price * (1 - gap_pct / 100.0)
            else: # sell
                candidate_stop = peak_price * (1 + gap_pct / 100.0)

            current_stop = self.last_stop_price.get(symbol, 0.0)
            is_better = (side == "buy" and candidate_stop > current_stop) or \
                        (side == "sell" and (current_stop == 0.0 or candidate_stop < current_stop))

            if is_better:
                # не кладём стоп через рынок: подожмём его к текущей цене на 1 тик
                tick = float(self.price_tick_map.get(symbol, 0.0) or 1e-6)

                if side == "buy" and candidate_stop >= last_price:
                    candidate_stop = last_price - tick
                elif side == "sell" and candidate_stop <= last_price:
                    candidate_stop = last_price + tick

                # отправляем один раз (без дублирования)
                self.last_trailing_update_ts[symbol] = now
                await self.set_or_amend_stop_loss(candidate_stop, symbol=symbol)

                prev_stop = self._trailing_prev_stop.get(symbol)
                last_log_ts = self._trailing_log_ts.get(symbol, 0.0)
                tick_diff = 0.0
                if prev_stop is not None and tick > 0:
                    tick_diff = abs(candidate_stop - prev_stop) / tick

                if (
                    prev_stop is None
                    or tick_diff >= min_log_delta
                    or now - last_log_ts >= log_throttle
                ):
                    logger.info(
                        f"📈 [{symbol}] Двигаю стоп (простой гэп). "
                        f"Пик: {peak_price:.6f}, Новый SL: {candidate_stop:.6f}"
                    )
                    self._trailing_log_ts[symbol] = now

                self._trailing_prev_stop[symbol] = candidate_stop


    async def _run_dynamic_atr_trailing(self, symbol: str, last_price: float, pos: dict):
        """
        Двухфазный трейлинг:
        1) PRE-фаза до целевого ROI (с учётом плеча): стоп от ближайшей противоположной DOM-плотности с запасом k_pre*ATR.
        2) После активации (ROI_lev ≥ TRAIL_ACTIVATE_ROI_LEVERED_PCT): динамический ATR/ADX трейлинг (fade+wall) + ratchet.
        Всегда соблюдаем биржевые правила (SL по правильную сторону и мин. тик-отступ).
        """
        self._ensure_trailing_state(symbol)
        tcfg = config.TRAILING_MODES.get("dynamic", {})

        # --- защита от частых апдейтов
        now = time.time()
        if now - self.last_trailing_update_ts.get(symbol, 0.0) < float(tcfg.get("MIN_UPDATE_INTERVAL_SECS", 0.3)):
            return

        lock = self.trailing_lock[symbol]
        if lock.locked():
            return

        async with lock:
            pos = self.open_positions.get(symbol)
            if not pos:
                return

            side = (pos.get("side", "Buy") or "Buy").lower()
            avg_price = utils.safe_to_float(pos.get("avgPrice") or pos.get("avg_price") or 0.0)
            if avg_price <= 0.0 or last_price <= 0.0:
                return

            # тайминг позиции
            opened_ts   = utils.safe_to_float(pos.get("open_ts") or pos.get("createdTs") or 0.0)
            time_in_pos = (now - opened_ts) if opened_ts > 0 else 0.0

            # тик
            tick = float(self.price_tick_map.get(symbol, 0.0) or 0.0) or 1e-6

            # peak по направлению позиции
            if "trailing_peak" not in pos:
                pos["trailing_peak"] = avg_price
            if side == "buy":
                pos["trailing_peak"] = max(pos["trailing_peak"], last_price)
            else:
                pos["trailing_peak"] = min(pos["trailing_peak"], last_price)
            peak = float(pos["trailing_peak"])

            # фичи
            features = await self.extract_realtime_features(symbol)
            if not features:
                return
            adx     = float(features.get("adx14", 0.0))
            atr_val = float(features.get("atr14", 0.0))
            if atr_val <= 0.0:
                return

            # эффективное плечо
            lev = utils.safe_to_float(
                pos.get("leverage")
                or getattr(self, "symbol_leverage_map", {}).get(symbol)
                or tcfg.get("ASSUME_LEVERAGE_IF_MISSING", 1.0)
                or 1.0
            )
            lev = max(1.0, lev)

            # ROI: считаем и «ценовой» (без плеча), и «левереджный», берём только профитную часть
            roi_unlev_signed = (last_price - avg_price) / max(1e-12, avg_price) * 100.0
            if side == "sell":
                roi_unlev_signed = (avg_price - last_price) / max(1e-12, avg_price) * 100.0
            roi_unlev_profit = max(0.0, roi_unlev_signed)  # % хода цены
            roi_lev_profit   = roi_unlev_profit * lev      # твой ROI «с плечом»

            # === ФАЗА 1: PRE до активации по целевому ROI (С УЧЁТОМ ПЛЕЧА) ===
            roi_target_lev = float(tcfg.get("TRAIL_ACTIVATE_ROI_LEVERED_PCT", 5.0))
            if roi_lev_profit < roi_target_lev:
                # стоп от ближайшей ПРОТИВОПОЛОЖНОЙ стены с запасом k_pre*ATR
                k_pre = float(tcfg.get("PRE_WALL_ATR_K", 3.0))
                wall_price = None

                get_opp = getattr(self, "get_dom_nearest_opposite_wall_price", None)
                getter  = getattr(self, "get_dom_next_wall_price", None)
                try:
                    if callable(get_opp):
                        maybe = get_opp(symbol, side)
                        wall_price = await maybe if inspect.iscoroutine(maybe) else maybe
                    elif callable(getter):
                        opposite = "sell" if side == "buy" else "buy"
                        maybe = getter(symbol, opposite)
                        wall_price = await maybe if inspect.iscoroutine(maybe) else maybe
                except Exception:
                    wall_price = None

                if wall_price:
                    candidate = wall_price - k_pre * atr_val if side == "buy" else wall_price + k_pre * atr_val
                    src = f"wall±{k_pre}*ATR"
                else:
                    candidate = avg_price - k_pre * atr_val if side == "buy" else avg_price + k_pre * atr_val
                    src = f"avg±{k_pre}*ATR (fallback)"

                # Биржевой хард-лимит + мин. отступ
                min_gap_ticks = int(tcfg.get("MIN_GAP_TICKS", 2))
                best_bid = utils.safe_to_float(self.best_bid_map.get(symbol, 0.0))
                best_ask = utils.safe_to_float(self.best_ask_map.get(symbol, 0.0))
                ref_buy  = min([v for v in (last_price, best_bid) if v]) if (last_price or best_bid) else last_price
                ref_sell = max([v for v in (last_price, best_ask) if v]) if (last_price or best_ask) else last_price

                if side == "buy":
                    hard_floor = ref_buy - min_gap_ticks * tick
                    if candidate >= hard_floor:
                        candidate = hard_floor - tick
                else:
                    hard_ceiling = ref_sell + min_gap_ticks * tick
                    if candidate <= hard_ceiling:
                        candidate = hard_ceiling + tick

                candidate = self._round_to_tick(symbol, float(candidate), "buy" if side == "buy" else "sell")

                # ratchet: не ухудшаем
                last_sent = float(self.last_sent_stop_price.get(symbol, 0.0) or 0.0)
                if last_sent > 0.0:
                    candidate = max(candidate, last_sent) if side == "buy" else min(candidate, last_sent)

                logger.debug(
                    f"🛡️ [{symbol}] PRE-TRAIL до {roi_target_lev:.2f}% ROI(@x{lev:.1f}): wall={wall_price}, "
                    f"ATR={atr_val:.6f}, {src}, SL→{candidate:.6f} | roi_price={roi_unlev_profit:.2f}% "
                    f"(ROI_lev={roi_lev_profit:.2f}%), t={time_in_pos:.1f}s"
                )

                self.last_trailing_update_ts[symbol] = now
                await self.set_or_amend_stop_loss(candidate, symbol=symbol)
                return

            # === ФАЗА 2: ДИНАМИЧЕСКИЙ ATR/ADX ТРЕЙЛИНГ ===

            # ROI от пика (по цене, без плеча) — для тиеров k
            roi_peak_unlev = abs(peak - avg_price) / max(1e-12, avg_price) * 100.0

            def pick_k(adx_val: float, roi_val_unlev: float) -> float:
                tiers = tcfg.get("ROI_TIERS", [])
                elig = [t for t in tiers if roi_val_unlev >= float(t.get("roi", 0.0))]
                if not elig:
                    return float(tcfg.get("K_DEFAULT", 2.5))
                top_roi = max(e.get("roi", 0.0) for e in elig)
                cand = [e for e in elig if e.get("roi", 0.0) == top_roi]

                def band_match(b: str, a: float) -> bool:
                    return (
                        (b == "adx_gt_30" and a > 30.0)
                        or (b == "adx_ge_20" and 20.0 <= a <= 30.0)
                        or (b == "adx_lt_20" and a < 20.0)
                    )
                for e in cand:
                    if band_match(str(e.get("band", "")), adx_val):
                        return float(e.get("k", tcfg.get("K_DEFAULT", 2.5)))
                return float(cand[0].get("k", tcfg.get("K_DEFAULT", 2.5))) if cand else float(tcfg.get("K_DEFAULT", 2.5))

            k_base = pick_k(adx, roi_peak_unlev)

            # против потока — ужесточаем
            cvd5m = float(features.get("CVD5m", 0.0))
            if (side == "buy" and cvd5m < 0.0) or (side == "sell" and cvd5m > 0.0):
                k_base *= float(tcfg.get("FLOW_TIGHTEN_FACTOR", 0.85))

            # Армирование безубытка (по цене, без плеча)
            be_buf_pct     = float(tcfg.get("BREAKEVEN_BUFFER_PCT", 0.18))
            be_arm_sec     = float(tcfg.get("BREAKEVEN_ARM_SEC", 20.0))
            be_arm_roi_pct = float(tcfg.get("BREAKEVEN_ARM_ROI_PCT", 0.25))
            be_prev        = bool(pos.get("be_armed", False))
            be_now         = (time_in_pos >= be_arm_sec) or (roi_unlev_profit >= be_arm_roi_pct)
            if be_now and not be_prev:
                pos["be_armed"] = True
                logger.info(f"[{symbol}] BE-ARM: t={time_in_pos:.1f}s, roi_price={roi_unlev_profit:.2f}% → активирован безубыток.")
            be_armed = be_prev or be_now

            # ADX-затухание (история создана в _ensure_trailing_state)
            dq = self._adx_hist.get(symbol)
            dq.append((now, adx))
            adx_slope_pm = 0.0
            if len(dq) >= 2:
                t0, a0 = dq[0]
                t1, a1 = dq[-1]
                dt = max(1e-6, t1 - t0)
                adx_slope_pm = (a1 - a0) / dt * 60.0

            slowdown_slope = abs(float(tcfg.get("SLOWDOWN_ADX_SLOPE_PM", 6.0)))
            slowdown_decay = float(tcfg.get("SLOWDOWN_ROI_DECAY_SEC", 20.0))
            slowdown_weight = float(tcfg.get("SLOWDOWN_ACCEL_WEIGHT", 0.6))
            slowdown_min_gap = max(1, int(tcfg.get("SLOWDOWN_MIN_GAP_TICKS", 1)))
            peak_age = max(0.0, now - pos.get("trailing_peak_ts", now))
            slowdown_factor = 0.0
            if slowdown_slope > 0 and adx_slope_pm < 0:
                slowdown_factor = max(slowdown_factor, min(1.0, abs(adx_slope_pm) / slowdown_slope))
            if slowdown_decay > 0 and peak_age > slowdown_decay:
                slowdown_factor = max(slowdown_factor, min(1.0, (peak_age - slowdown_decay) / slowdown_decay))
            slowdown_factor = min(1.0, slowdown_factor)

            fade = 0.0
            if adx < float(tcfg.get("FADE_ADX_LT", 18.0)):
                fade += float(tcfg.get("FADE_WEIGHT_LOW_ADX", 0.5))
            if adx_slope_pm < -abs(float(tcfg.get("FADE_ADX_SLOPE_DOWN_PM", 4.0))):
                fade += float(tcfg.get("FADE_WEIGHT_SLOPE", 0.5))
            fade = min(1.0, max(0.0, fade))

            stall_after = float(tcfg.get("STALL_TIGHTEN_AFTER_SEC", 0.0))
            stall_step_sec = max(1.0, float(tcfg.get("STALL_TIGHTEN_STEP_SEC", 10.0)))
            stall_step_pct = float(tcfg.get("STALL_TIGHTEN_STEP_PCT", 0.0)) / 100.0
            stall_max = float(tcfg.get("STALL_TIGHTEN_MAX", 0.0))
            if (
                stall_after > 0.0
                and stall_step_pct > 0.0
                and stall_max > 0.0
                and self.trailing_activated.get(symbol, False)
                and roi_unlev_profit > 0.0
                and peak_age > stall_after
            ):
                stall_steps = int((peak_age - stall_after) / stall_step_sec) + 1
                tighten_ratio = min(stall_max, stall_steps * stall_step_pct)
                tb *= max(0.1, 1.0 - tighten_ratio)

            # «застревание» у следующей стены по направлению позиции
            u_wall = 0.0
            next_wall_price = None
            getter = getattr(self, "get_dom_next_wall_price", None)
            try:
                if callable(getter):
                    maybe = getter(symbol, side)
                    next_wall_price = await maybe if inspect.iscoroutine(maybe) else maybe
            except Exception:
                next_wall_price = None

            if next_wall_price:
                band_ticks = int(tcfg.get("WALL_BAND_TICKS", 4))
                if side == "buy":
                    dist_ticks = int(round((next_wall_price - last_price) / tick))
                else:
                    dist_ticks = int(round((last_price - next_wall_price) / tick))
                if 0 <= dist_ticks <= band_ticks:
                    since = self._wall_stall_since.get(symbol)
                    if not since:
                        self._wall_stall_since[symbol] = now
                        since = now
                    stall_sec = now - since
                    stall_thr = float(tcfg.get("WALL_STALL_SEC", 8.0))
                    if stall_sec >= stall_thr:
                        u_wall = min(1.0, (stall_sec - stall_thr) / max(1.0, float(tcfg.get("WALL_STALL_MAX_EXTRA", 10.0))))
                else:
                    self._wall_stall_since[symbol] = None

            # интегральная срочность
            u = (
                float(tcfg.get("FADE_URGENCY_WEIGHT", 0.6)) * fade
                + float(tcfg.get("WALL_URGENCY_WEIGHT", 0.6)) * u_wall
            )
            u = min(float(tcfg.get("MAX_TIGHTEN_URGENCY", 0.9)), max(0.0, u))

            # уменьшаем k при росте срочности
            k = max(float(tcfg.get("K_MIN", 0.8)), k_base * (1.0 - u * float(tcfg.get("URGENCY_K_SHRINK", 0.6))))

            # кандидат по пику
            peak_stop = (peak - atr_val * k) if side == "buy" else (peak + atr_val * k)

            # безубыток (только если «вооружён»)
            if be_armed:
                if side == "buy":
                    be_price = avg_price * (1.0 + be_buf_pct / 100.0)
                    candidate = max(peak_stop, be_price)
                else:
                    be_price = avg_price * (1.0 - be_buf_pct / 100.0)
                    candidate = min(peak_stop, be_price)
            else:
                candidate = peak_stop

            # плавающее ускорение к цене + при затухании прижимаемся к пику
            min_gap_ticks   = int(tcfg.get("MIN_GAP_TICKS", 2))
            urg_gap_ticks   = int(tcfg.get("URGENCY_EXTRA_GAP_TICKS", 1))
            total_gap_ticks = max(1, min_gap_ticks + int(round(u * urg_gap_ticks)))
            peak_lock_ticks = int(tcfg.get("PEAK_LOCK_TICKS_MIN", min_gap_ticks))
            peak_lock_atr   = max(0.0, float(tcfg.get("PEAK_LOCK_ATR_MULT", 0.45)))
            fade_to_peak_w  = min(1.0, max(0.0, fade) * float(tcfg.get("FADE_TO_PEAK_WEIGHT", 1.0)))

            if side == "buy":
                price_floor = last_price - total_gap_ticks * tick
                candidate = (1.0 - u) * candidate + u * price_floor

                hard_limit = last_price - min_gap_ticks * tick
                if slowdown_factor > 0.0:
                    accel_gap_ticks = max(slowdown_min_gap, 1)
                    accel_target = last_price - accel_gap_ticks * tick
                    accel_target = min(accel_target, hard_limit - tick)
                    accel_target = max(accel_target, price_floor)
                    candidate = candidate + slowdown_weight * slowdown_factor * (accel_target - candidate)
                if fade_to_peak_w > 0.0:
                    peak_gap = max(peak_lock_ticks * tick, peak_lock_atr * atr_val)
                    peak_target = peak - peak_gap
                    peak_target = min(peak_target, hard_limit - tick)
                    if peak_target > 0.0:
                        candidate = (1.0 - fade_to_peak_w) * candidate + fade_to_peak_w * peak_target

                if candidate >= hard_limit:
                    candidate = hard_limit - tick
                candidate = max(candidate, price_floor)
            else:
                price_ceiling = last_price + total_gap_ticks * tick
                candidate = (1.0 - u) * candidate + u * price_ceiling

                hard_limit = last_price + min_gap_ticks * tick
                if slowdown_factor > 0.0:
                    accel_gap_ticks = max(slowdown_min_gap, 1)
                    accel_target = last_price + accel_gap_ticks * tick
                    accel_target = max(accel_target, hard_limit + tick)
                    accel_target = min(accel_target, price_ceiling)
                    candidate = candidate + slowdown_weight * slowdown_factor * (accel_target - candidate)
                if fade_to_peak_w > 0.0:
                    peak_gap = max(peak_lock_ticks * tick, peak_lock_atr * atr_val)
                    peak_target = peak + peak_gap
                    peak_target = max(peak_target, hard_limit + tick)
                    candidate = (1.0 - fade_to_peak_w) * candidate + fade_to_peak_w * peak_target

                if candidate <= hard_limit:
                    candidate = hard_limit + tick
                candidate = min(candidate, price_ceiling)

            # биржевые лимиты
            best_bid = utils.safe_to_float(self.best_bid_map.get(symbol, 0.0))
            best_ask = utils.safe_to_float(self.best_ask_map.get(symbol, 0.0))
            ref_buy  = min([v for v in (last_price, best_bid) if v]) if (last_price or best_bid) else last_price
            ref_sell = max([v for v in (last_price, best_ask) if v]) if (last_price or best_ask) else last_price
            if side == "buy":
                hard_floor = ref_buy - min_gap_ticks * tick
                if candidate >= hard_floor:
                    candidate = hard_floor - tick
            else:
                hard_ceiling = ref_sell + min_gap_ticks * tick
                if candidate <= hard_ceiling:
                    candidate = hard_ceiling + tick

            candidate = self._round_to_tick(symbol, float(candidate), "buy" if side == "buy" else "sell")

            if side == "buy":
                hard_limit = ref_buy - min_gap_ticks * tick
                price_floor = last_price - total_gap_ticks * tick
                if candidate >= hard_limit:
                    candidate = hard_limit - tick
                candidate = max(candidate, price_floor)
            else:
                hard_limit = ref_sell + min_gap_ticks * tick
                price_ceiling = last_price + total_gap_ticks * tick
                if candidate <= hard_limit:
                    candidate = hard_limit + tick
                candidate = min(candidate, price_ceiling)

            # ratchet (не ухудшаем ранее отправленный)
            last_sent = float(self.last_sent_stop_price.get(symbol, 0.0) or 0.0)
            if last_sent > 0.0:
                candidate = max(candidate, last_sent) if side == "buy" else min(candidate, last_sent)

            # лог (антиспам)
            prev_logged = self._trailing_prev_stop.get(symbol)
            t_last_log  = self._trailing_log_ts.get(symbol, 0.0)
            moved_one_tick = (prev_logged is None or abs(candidate - (prev_logged or candidate)) >= tick)
            time_ok = (now - t_last_log) >= float(tcfg.get("MIN_LOG_INTERVAL_SECS", 12.0))
            if moved_one_tick or time_ok:
                logger.info(
                    f"📈 [{symbol}] TRAIL ATR-ADX: peak={peak:.6f}, ATR={atr_val:.6f}, "
                    f"k_base={k_base:.2f}, k={k:.2f}, SL→{candidate:.6f} | "
                    f"roi_price={roi_unlev_profit:.2f}% (ROI_lev={roi_lev_profit:.2f}%), "
                    f"t={time_in_pos:.1f}s, fade={fade:.2f}, wallU={u_wall:.2f}, u={u:.2f}, be_armed={be_armed}"
                )
                self._trailing_prev_stop[symbol] = candidate
                self._trailing_log_ts[symbol] = now

            self.last_trailing_update_ts[symbol] = now
            await self.set_or_amend_stop_loss(candidate, symbol=symbol)

            

    def _sl_api_cooldown_ok(self, symbol: str, min_interval: float = 1.7) -> bool:
        import time
        last = 0.0 if not hasattr(self, "_sl_api_last_ts") else self._sl_api_last_ts.get(symbol, 0.0)
        now = time.time()
        if now - last < float(min_interval):
            return False
        if not hasattr(self, "_sl_api_last_ts"):
            self._sl_api_last_ts = {}
        self._sl_api_last_ts[symbol] = now
        return True

    async def set_or_amend_stop_loss_throttled(self, price: float, *, symbol: str):
        """
        Обёртка над твоим set_or_amend_stop_loss с пер-символьным API-кулдауном и фиксацией last_sent_stop_price.
        """
        if not self._sl_api_cooldown_ok(symbol, 1.7):
            return
        await self.set_or_amend_stop_loss(price, symbol=symbol)
        # если дошли до сюда — API принял команду: фиксируем «последний отправленный»
        if not hasattr(self, "last_sent_stop_price"):
            self.last_sent_stop_price = {}
        self.last_sent_stop_price[symbol] = float(price)


    async def get_total_open_volume(self) -> float:
        total = 0.0
        for pos in self.open_positions.values():
            size = utils.safe_to_float(pos.get("volume", 0))
            price = utils.safe_to_float(pos.get("markPrice", 0)) or utils.safe_to_float(pos.get("avg_price", 0))
            total += size * price
        return total

    async def ensure_recent_candles(self, symbol: str, lookback: int = 120, max_age_sec: float = 180.0) -> None:
        ensure = getattr(self.shared_ws, "ensure_recent_candles", None)
        if callable(ensure):
            try:
                await ensure(symbol, lookback=lookback, max_age_sec=max_age_sec)
            except Exception as e:
                logger.warning(f"[{symbol}] Не удалось обновить свечи: {e}")

    def _calc_open_positions_notional(self) -> float:
        total = 0.0
        try:
            for symbol, pos_data in self.open_positions.items():
                size = utils.safe_to_float(pos_data.get("size") or pos_data.get("volume", 0))
                price = utils.safe_to_float(
                    pos_data.get("avg_price")
                    or pos_data.get("avgPrice")
                    or pos_data.get("markPrice", 0)
                )
                if size > 0 and price > 0:
                    notional = abs(size) * price
                    total += notional
                    logger.debug(f"[VOLUME_CALC] {symbol}: номинал = {notional:.2f} USDT")
        except Exception as e:
            logger.warning(f"[VOLUME_CALC] Ошибка подсчёта по открытым позициям: {e}", exc_info=True)
        return total

    async def get_effective_total_volume(self) -> float:
        """
        Возвращает суммарный текущий объём позиций и зарезервированных входов
        в долларовом эквиваленте (без деления на плечо).
        """
        total_notional = self._calc_open_positions_notional()

        try:
            async with self.pending_orders_lock:
                pending_notional = sum(abs(v) for v in self.pending_orders.values())
            total_notional += pending_notional
            if pending_notional:
                logger.debug(f"[VOLUME_CALC] Pending-номинал: {pending_notional:.2f} USDT")
        except Exception as e:
            logger.error(f"[VOLUME_CALC] Ошибка подсчёта pending-номинала: {e}", exc_info=True)

        logger.info(f"[VOLUME_CALC] Общий номинал позиций: {total_notional:.2f} USDT")
        return total_notional



    @async_retry(max_retries=5, delay=3)
    async def _cache_all_symbol_meta(self):
        logger.info("Кэширование метаданных для всех символов...")
        try:
            resp = await asyncio.to_thread(lambda: self.session.get_instruments_info(category="linear"))
            instrument_list = resp.get("result", {}).get("list", [])
            for info in instrument_list:
                if symbol := info.get("symbol"):
                    lot_filter = info.get("lotSizeFilter", {}) or {}
                    price_filter = info.get("priceFilter", {}) or {}
                    qty_step = utils.safe_to_float(lot_filter.get("qtyStep"))
                    self.qty_step_map[symbol] = qty_step
                    self.min_qty_map[symbol] = utils.safe_to_float(lot_filter.get("minOrderQty"))
                    raw_max = lot_filter.get("maxOrderQty")
                    if raw_max is not None and qty_step and qty_step > 0:
                        try:
                            max_qty_decimal = Decimal(str(raw_max)) * Decimal(str(qty_step))
                            self.max_qty_map[symbol] = float(max_qty_decimal)
                        except Exception:
                            self.max_qty_map[symbol] = utils.safe_to_float(raw_max)
                    else:
                        self.max_qty_map[symbol] = utils.safe_to_float(raw_max)
                    self.price_tick_map[symbol] = utils.safe_to_float(price_filter.get("tickSize"))
            logger.info(f"Успешно закэшировано метаданных для {len(self.qty_step_map)} символов.")
        except Exception:
            logger.error("Критическая ошибка: не удалось закэшировать метаданные символов.", exc_info=True)

    def load_ml_models(self):
        self.ml_inferencer = ai_ml.MLXInferencer(
            model_path=config.ML_MODEL_PATH,
            scaler_path=config.SCALER_PATH,
        )

    async def extract_realtime_mkt_features(self, symbol: str) -> dict:
        """
        Совместимость для старых вызовов стратегий.
        """
        try:
            feats = await self.extract_realtime_features(symbol)
            return feats if isinstance(feats, dict) else {}
        except Exception:
            return {}


    def _extract_realtime_features_sync(self, symbol: str) -> Optional[Dict[str, float]]:
        def _safe_last(series, default):
            if series is None or not isinstance(series, pd.Series) or series.empty: return default
            try:
                v = series.iloc[-1]
                return v if pd.notna(v) else default
            except IndexError: return default
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
        flea_cfg = self.user_data.get("flea_settings", config.FLEA_STRATEGY)
        fast_ema_len, slow_ema_len, trend_ema_len = flea_cfg.get("FAST_EMA_PERIOD", 5), flea_cfg.get("SLOW_EMA_PERIOD", 10), flea_cfg.get("TREND_EMA_PERIOD", 200)
        fast_ema_val = _safe_last(ta.ema(close, length=fast_ema_len), 0.0) if n >= fast_ema_len else 0.0
        slow_ema_val = _safe_last(ta.ema(close, length=slow_ema_len), 0.0) if n >= slow_ema_len else 0.0
        trend_ema_val = _safe_last(ta.ema(close, length=trend_ema_len), 0.0) if n >= trend_ema_len else 0.0
        fast_ema_prev = _safe_last(ta.ema(close, length=fast_ema_len).shift(1), 0.0) if n > fast_ema_len else 0.0
        slow_ema_prev = _safe_last(ta.ema(close, length=slow_ema_len).shift(1), 0.0) if n > slow_ema_len else 0.0
        avg_volume_prev_4m = 0
        if len(candles) >= 5:
            avg_volume_prev_4m = np.mean([utils.safe_to_float(c.get("volume", 0)) for c in candles[-5:-1]])
        rsi14 = _safe_last(ta.rsi(close, length=14), 50.0) if n >= 15 else 50.0
        sma50 = _safe_last(ta.sma(close, length=50), _safe_last(close, 0.0)) if n >= 50 else _safe_last(close, 0.0)
        ema20 = _safe_last(ta.ema(close, length=20), sma50) if n >= 20 else sma50
        atr_series = ta.atr(high, low, close, length=14) if n >= 15 else None
        atr14 = _safe_last(atr_series, 0.0) if atr_series is not None else 0.0
        if (atr14 <= 0.0 or np.isnan(atr14)) and n >= 2:
            try:
                high_np = high.to_numpy(dtype="float64", copy=True)
                low_np = low.to_numpy(dtype="float64", copy=True)
                close_np = close.to_numpy(dtype="float64", copy=True)
                prev_close = close_np[:-1]
                curr_high = high_np[1:]
                curr_low = low_np[1:]
                tr1 = curr_high - curr_low
                tr2 = np.abs(curr_high - prev_close)
                tr3 = np.abs(curr_low - prev_close)
                true_ranges = np.maximum.reduce([tr1, tr2, tr3])
                if true_ranges.size:
                    window = true_ranges[-14:] if true_ranges.size >= 14 else true_ranges
                    atr_manual = float(np.mean(window))
                    if atr_manual > 0:
                        atr14 = atr_manual
                        logger.debug(f"[features] {symbol}: ATR14 восстановлен вручную ({true_ranges.size} TR выборок).")
            except Exception as e:
                logger.debug(f"[features] {symbol}: не удалось вычислить ATR fallback: {e}")
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
        hour_of_day, day_of_week, month_of_year = int(now_ts.hour), int(now_ts.weekday()), int(now_ts.month)
        avgVol30m = self.shared_ws.get_avg_volume(symbol, 30)
        baseline_vol = 0.0
        if avgVol30m and avgVol30m > 0:
            baseline_vol = float(avgVol30m) / 30.0
        if baseline_vol <= 0 and len(candles) >= 30:
            last_30_vol = [utils.safe_to_float(c.get("volume", 0.0)) for c in candles[-30:]]
            baseline_vol = float(np.nanmean(last_30_vol)) if last_30_vol else 0.0
        volume_anomaly = float(V1m) / baseline_vol if baseline_vol > 0 else 1.0
        volume_anomaly = float(np.clip(volume_anomaly, 0.0, 1000.0))
        tail_oi = [utils.safe_to_float(x) for x in oi_hist[-30:]] if oi_hist else []
        avgOI30m = float(np.nanmean(tail_oi)) if tail_oi else 0.0
        deltaCVD30m = CVD_now - (utils.safe_to_float(cvd_hist[-31]) if len(cvd_hist) >= 31 else 0.0)
        candles_15m = self._aggregate_candles_15m(candles)
        df_15m = pd.DataFrame(candles_15m)
        atr15m = 0.0
        if len(df_15m) >= 15:
            high15, low15, close15 = pd.to_numeric(df_15m["highPrice"]), pd.to_numeric(df_15m["lowPrice"]), pd.to_numeric(df_15m["closePrice"])
            atr_series_15 = ta.atr(high15, low15, close15, length=14)
            if atr_series_15 is not None and not atr_series_15.empty:
                atr15m = _safe_last(atr_series_15, 0.0)
        candles_1h = self._aggregate_candles_60m(candles)
        trend_h1 = 0
        if len(candles_1h) > 2:
            if candles_1h[-1]['closePrice'] > candles_1h[-2]['closePrice']: trend_h1 = 1
            elif candles_1h[-1]['closePrice'] < candles_1h[-2]['closePrice']: trend_h1 = -1
        funding_snap = self._funding_snapshot(symbol, tdata) 
        features: Dict[str, float] = {
            "price": last_price, "pct1m": pct1m, "pct5m": pct5m, "pct15m": pct15m, "vol1m": V1m, "vol5m": V5m, "vol15m": V15m,
            "OI_now": OI_now, "dOI1m": dOI1m, "dOI5m": dOI5m, "spread_pct": spread_pct, "sigma5m": sigma5m, "CVD1m": CVD1m, "CVD5m": CVD5m,
            "rsi14": rsi14, "sma50": sma50, "ema20": ema20, "atr14": atr14, "bb_width": bb_width, "supertrend": supertrend_num,
            "cci20": cci20, "macd": macd_val, "macd_signal": macd_signal, "avgVol30m": avgVol30m, "avgOI30m": avgOI30m,
            "deltaCVD30m": deltaCVD30m, "adx14": adx14, "GS_pct4m": GS_pct4m, "GS_vol4m": GS_vol4m, "GS_dOI4m": GS_dOI4m,
            "GS_cvd4m": GS_cvd4m, "GS_supertrend": GS_supertrend_flag, "GS_cooldown": GS_cooldown_flag, "SQ_pct1m": pct1m,
            "SQ_pct5m": pct5m, "SQ_vol1m": V1m, "SQ_vol5m": V5m, "SQ_dOI1m": dOI1m, "SQ_spread_pct": spread_pct, "SQ_sigma5m": sigma5m,
            "SQ_liq10s": SQ_liq10s, "SQ_cooldown": SQ_cooldown_flag, "SQ_power": SQ_power, "SQ_strength": SQ_strength,
            "LIQ_cluster_val10s": LIQ_cluster_val10s, "LIQ_cluster_count10s": LIQ_cluster_count10s, "LIQ_direction": LIQ_direction,
            "LIQ_pct1m": pct1m, "LIQ_pct5m": pct5m, "pct30m": pct30m, "LIQ_vol1m": V1m, "LIQ_vol5m": V5m, "LIQ_dOI1m": dOI1m,
            "LIQ_spread_pct": spread_pct, "LIQ_sigma5m": sigma5m, "LIQ_golden_flag": GS_cooldown_flag, "LIQ_squeeze_flag": SQ_cooldown_flag,
            "LIQ_cooldown": LIQ_cooldown_flag, "hour_of_day": hour_of_day, "day_of_week": day_of_week, "month_of_year": month_of_year,
            "atr15m": atr15m, "trend_h1": trend_h1, "funding_rate": funding_snap.get("funding_rate", 0.0), "fast_ema": fast_ema_val,
            "slow_ema": slow_ema_val, "trend_ema": trend_ema_val, "fast_ema_prev": fast_ema_prev, "slow_ema_prev": slow_ema_prev,
            "avg_volume_prev_4m": avg_volume_prev_4m, "volume_anomaly": volume_anomaly,
        }
        for k in config.FEATURE_KEYS: features.setdefault(k, 0.0)
        if isinstance(features, dict):
            if "last_price" in features and "lastPrice" not in features:
                features["lastPrice"] = features["last_price"]
            if "mark_price" in features and "markPrice" not in features:
                features["markPrice"] = features["mark_price"]
        features["last_price"] = utils.safe_to_float(last_price)
        features["lastPrice"]  = utils.safe_to_float(last_price)
        features["markPrice"]  = utils.safe_to_float(last_price)

        return features

    def get_last_price(self, symbol: str) -> float:
        """
        Возвращает last price из доступных источников.
        """
        try:
            tdata = getattr(self.shared_ws, "ticker_data", {}).get(symbol) or {}
        except Exception:
            tdata = {}

        candidates = [
            tdata.get("lastPrice"), tdata.get("markPrice"), tdata.get("last_price"),
            tdata.get("price"), tdata.get("indexPrice"),
        ]

        for v in candidates:
            x = utils.safe_to_float(v)
            if x and x > 0:
                return x
        return 0.0


    async def extract_realtime_features(self, symbol: str) -> Optional[Dict[str, float]]:
        # Проверяем кэш
        current_time = time.time()
        cache_key = symbol
        if cache_key in self._feature_cache:
            cached_data, timestamp = self._feature_cache[cache_key]
            if current_time - timestamp < self._feature_cache_ttl:
                return cached_data
        
        async with self.feature_extraction_sem:
            features = await asyncio.to_thread(self._extract_realtime_features_sync, symbol)
            # Сохраняем в кэш
            if features:
                self._feature_cache[cache_key] = (features, current_time)
            return features

    async def _cleanup_feature_cache(self):
        """Очистка устаревших записей из кэша features"""
        while True:
            try:
                await asyncio.sleep(30)  # Очищаем каждые 30 секунд
                current_time = time.time()
                expired_keys = [
                    key for key, (_, timestamp) in self._feature_cache.items()
                    if current_time - timestamp > self._feature_cache_ttl * 2
                ]
                for key in expired_keys:
                    del self._feature_cache[key]
                if expired_keys:
                    logger.debug(f"Очищено {len(expired_keys)} устаревших записей из кэша features")
            except Exception as e:
                logger.error(f"Ошибка очистки кэша features: {e}")

    async def _get_golden_thresholds(self, symbol: str, side: str) -> dict:
        base = (self.golden_param_store.get((symbol, side)) or self.golden_param_store.get(side) or 
                {"period_iters": 3, "price_change": 1.7, "volume_change": 200, "oi_change": 1.5})
        return base

    def _aggregate_candles_5m(self, candles: any) -> list:
        candle_list = list(candles)
        if not candle_list: return []
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
            if method == "sum": result.append(sum(utils.safe_to_float(x) for x in chunk))
            else: result.append(utils.safe_to_float(chunk[-1]))
        return result[-lookback:]

    async def update_orderbook_metrics(self, symbol: str, metrics: dict):
        if not metrics:
            return
        self._orderbook_metrics[symbol] = metrics

    def _aggregate_ohlcv_5m(self, minute_candles: list, lookback: int = 15):
        try:
            if not minute_candles: return []
            m1_needed = lookback * 5
            tail = minute_candles[-m1_needed:] if len(minute_candles) >= m1_needed else minute_candles[:]
            bars_5m = []
            for i in range(0, len(tail), 5):
                chunk = tail[i:i+5]
                if len(chunk) < 5: break
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
            if len(bars) < 15: return None, None
            df = pd.DataFrame(bars)
            df['highPrice'] = pd.to_numeric(df['high'])
            df['lowPrice'] = pd.to_numeric(df['low'])
            df['closePrice'] = pd.to_numeric(df['close'])
            atr_series = ta.atr(df['highPrice'], df['lowPrice'], df['closePrice'], length=14)
            atr_5m = atr_series.iloc[-1] if not atr_series.empty and pd.notna(atr_series.iloc[-1]) else 0.0
            prev, last = bars[-2], bars[-1]
            pc = utils.safe_to_float(prev.get("close", 0.0))
            lc = utils.safe_to_float(last.get("close", 0.0))
            if pc <= 0 or lc <= 0: return None, None
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
        cooldown_period_sec = float(getattr(self, "golden_signal_cooldown_sec", 120.0))
        last_signal_time = self._last_golden_ts.get(symbol, 0)
        return (time.time() - last_signal_time) > cooldown_period_sec

    def _squeeze_allowed(self, symbol: str) -> bool:
        cooldown_period_sec = 600
        last_signal_time = self.last_squeeze_ts.get(symbol, 0)
        return (time.time() - last_signal_time) > cooldown_period_sec

    async def notify_user(self, text: str):
        if not telegram_bot: return
        last_error = None
        attempts = 5
        for attempt in range(attempts):
            try:
                await telegram_bot.send_message(self.user_id, text, parse_mode=ParseMode.HTML)
                return
            except Exception as e:
                last_error = e
                if attempt + 1 < attempts:
                    backoff = min(2 ** attempt, 30)
                    await asyncio.sleep(backoff)
        if last_error:
            logger.warning(f"Не удалось отправить сообщение пользователю {self.user_id}: {last_error}")

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
                if action == "open": self.flea_positions_count += 1
                elif action == "close": self.flea_positions_count = max(0, self.flea_positions_count - 1)
                logger.info(f"🦟 Счетчик позиций 'Блохи': {self.flea_positions_count}")
            elif action == "open":
                strategy_key = 'squeeze' if 'squeeze' in source.lower() else ('golden_setup' if 'golden' in source.lower() else None)
                if strategy_key:
                    self.trade_counters[strategy_key] += 1
                    logger.info(f"Счетчики основных стратегий обновлены: {dict(self.trade_counters)}")
            if action in {"open", "close"}:
                self._snapshot_open_positions()
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
            d_usd_amount, d_price, d_step, d_min_qty = Decimal(str(usd_amount)), Decimal(str(p)), Decimal(step_str), Decimal(min_qty_str)
            if d_price == 0: return 0.0
            raw_qty = d_usd_amount / d_price
            ticks = (raw_qty / d_step).quantize(Decimal('1'), rounding='ROUND_DOWN')
            final_qty = ticks * d_step
            if final_qty < d_min_qty:
                logger.warning(f"[{symbol}] Расчетный объем {final_qty} меньше минимального {d_min_qty}. Для ордера будет использован минимальный объем.")
                final_qty = d_min_qty

            max_qty = utils.safe_to_float(self.max_qty_map.get(symbol))
            if max_qty > 0 and final_qty > Decimal(str(max_qty)):
                max_qty_dec = Decimal(str(max_qty))
                capped_ticks = (max_qty_dec / d_step).quantize(Decimal('1'), rounding='ROUND_DOWN')
                capped_qty = capped_ticks * d_step
                if capped_qty <= 0:
                    logger.warning(f"[{symbol}] Биржевой максимум {max_qty} меньше минимального шага. Ордер отменён.")
                    return 0.0
                logger.warning(f"[{symbol}] Объем ограничен биржей: {final_qty} → {capped_qty} (max={max_qty}).")
                final_qty = capped_qty

            return float(final_qty)
        except Exception as e:
            logger.error(f"[{symbol}] Критическая ошибка при расчете количества: {e}", exc_info=True)
            return 0.0

    @async_retry(max_retries=5, delay=3)
    async def ensure_symbol_meta(self, symbol: str):
        if symbol in self.qty_step_map: return
        try:
            resp = await asyncio.to_thread(lambda: self.session.get_instruments_info(category="linear", symbol=symbol))
            info = resp["result"]["list"][0]
            lot_filter = info.get("lotSizeFilter", {}) or {}
            price_filter = info.get("priceFilter", {}) or {}

            qty_step = utils.safe_to_float(lot_filter.get("qtyStep"))
            min_qty_raw = lot_filter.get("minOrderQty")
            max_qty_raw = lot_filter.get("maxOrderQty")

            self.qty_step_map[symbol] = qty_step
            self.min_qty_map[symbol] = utils.safe_to_float(min_qty_raw)

            if max_qty_raw is not None and qty_step and qty_step > 0:
                try:
                    max_qty_decimal = Decimal(str(max_qty_raw)) * Decimal(str(qty_step))
                    self.max_qty_map[symbol] = float(max_qty_decimal)
                except Exception:
                    self.max_qty_map[symbol] = utils.safe_to_float(max_qty_raw)
            else:
                self.max_qty_map[symbol] = utils.safe_to_float(max_qty_raw)

            self.price_tick_map[symbol] = utils.safe_to_float(price_filter.get("tickSize"))
        except Exception as e:
            logger.warning(f"Не удалось получить метаданные для {symbol}: {e}")
            self.qty_step_map.setdefault(symbol, 0.001)
            self.min_qty_map.setdefault(symbol, 0.001)
            self.max_qty_map.setdefault(symbol, 0.0)
            self.price_tick_map.setdefault(symbol, 0.0001)
        
    async def listing_age_minutes(self, symbol: str) -> float:
        now = time.time()
        cached_data = _listing_age_cache.get(symbol)
        if cached_data and (now - cached_data[1] < 3600): return cached_data[0]
        async with _listing_sem:
            try:
                resp = await asyncio.to_thread(lambda: self.session.get_instruments_info(category="linear", symbol=symbol))
                info = resp["result"]["list"][0]
                launch_ms = utils.safe_to_float(info.get("launchTime", 0))
                if launch_ms <= 0: raise ValueError("launchTime missing or invalid")
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
        fm = {"funding_rate": funding_snap.get("funding_rate"), "funding_bucket": funding_snap.get("funding_bucket")}
        if "base_metrics" in candidate: candidate["base_metrics"].update(fm)
        else: candidate["base_metrics"] = fm

    def _funding_snapshot(self, symbol: str, features: dict | None = None) -> dict:
        rate = None
        if features: rate = features.get("fundingRate")
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
        if not features: return {}
        vol_1m = features.get('vol1m', 0)
        avg_vol_30m = features.get('avgVol30m', 1)
        vol_anomaly = vol_1m / avg_vol_30m if avg_vol_30m > 0 else 1.0
        metrics = {
            "price": utils.safe_to_float(features.get("price", 0.0)),
            "open_interest": utils.safe_to_float(features.get("OI_now", 0.0)),
            "volume_1m": utils.safe_to_float(vol_1m),
            "rsi14": utils.safe_to_float(features.get("rsi14", 0.0)),
            "adx14": utils.safe_to_float(features.get("adx14", 0.0)),
            "volume_anomaly": utils.safe_to_float(vol_anomaly),
        }
        # Добавляем полный вектор признаков, чтобы trades_unified.csv можно было использовать для обучения.
        detailed_features = {
            key: utils.safe_to_float(features.get(key, 0.0))
            for key in config.FEATURE_KEYS
        }
        metrics.update(detailed_features)
        return metrics
