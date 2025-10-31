# config.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# --- Основные настройки ---
ADMIN_IDS = {36972091}
BOT_DEVICE = "cpu"

# --- Пути к файлам ---
ROOT_DIR = Path(__file__).resolve().parent
LOG_FILE = ROOT_DIR / "bot.log"
WALLET_LOG_FILE = ROOT_DIR / "wallet_state.log"
HISTORY_FILE = ROOT_DIR / "history.pkl"
WALL_MEMORY_FILE = Path(__file__).parent / "wall_memory.pkl"
USER_STATE_FILE = ROOT_DIR / "user_state.json"
TREND_LOG_FILE = ROOT_DIR / "TREND_LOG.CSV"

# --- Пути для данных и моделей ---
ML_MODEL_PATH = ROOT_DIR / "golden_model_mlx.safetensors"
SCALER_PATH = ROOT_DIR / "scaler.pkl"
TRAINING_DATA_PATH = ROOT_DIR / "training_data.pkl"
ML_BEST_ENTRIES_CSV_PATH = ROOT_DIR / "ml_best_entries.csv"
SNAPSHOT_CSV_PATH = ROOT_DIR / "golden_setup_snapshots.csv"
LIQ_THRESHOLD_CSV_PATH = ROOT_DIR / "liq_thresholds.csv"
LIQUIDATIONS_CSV_PATH = ROOT_DIR / "liquidations.csv"
TRADES_UNIFIED_CSV_PATH = ROOT_DIR / "trades_unified.csv"
GOLDEN_PARAMS_CSV = ROOT_DIR / "golden_params.csv"
FINETUNE_LOG_FILE = ROOT_DIR / "finetune_log.csv"

# --- JSON-файлы для FSM ---
OPEN_POS_JSON = ROOT_DIR / "open_positions.json"
WALLET_JSON = ROOT_DIR / "wallet_state.json"
TRADES_JSON = ROOT_DIR / "trades_history.json"

# --- Параметры стратегий ---
SQUEEZE_THRESHOLD_PCT = 4.0
DEFAULT_SQUEEZE_POWER_MIN = 8.0
AVERAGE_LOSS_TRIGGER = -160.0
LISTING_AGE_MIN_MINUTES = 1400

# --- ML и AI ---
ML_CONFIDENCE_FLOOR = 0.65
ML_STRICT_FILTERS = {
    "ENABLED": True,
    "MIN_WORKING_ML": 0.33,
    "MAX_WORKING_ML": 0.85,
}

SIGNAL_MODEL = {
    "ENABLED": True,
    "MODEL_PATH": ROOT_DIR / "analysis/models/signal_model.pkl",
    "VECTORIZER_PATH": ROOT_DIR / "analysis/models/signal_vectorizer.pkl",
    "THRESHOLD": 0.75,
}

GOLDEN_PCT1M_FILTER = {
    "ENABLED": True,
    "MAX_PULLBACK_LONG": 0.18,   # В % (по факту 0.18%)
    "MAX_SURGE_SHORT": 0.18,
}

GOLDEN_CVD_FILTER = {
    "ENABLED": True,
    "MAX_CVD1M_LONG": 0.0,
    "MIN_CVD1M_SHORT": 0.0,
}

GOLDEN_ADX_FILTER = {
    "ENABLED": True,
    "MIN_ADX": 28.0,
}

GOLDEN_PRICE_EXHAUSTION_FILTER = {
    "ENABLED": True,
    "MAX_ABS_PCT5M": 2.5,   # Ограничиваем ход за последние ~5 минут
    "MAX_ABS_PCT15M": 6.0,  # Ограничение для более длинного окна
    "MAX_ABS_GS_PCT4M": 3.0,
}
AI_PRIMARY_MODEL = "trading-analyst"
AI_ADVISOR_MODEL = "0xroyce/plutus:latest"
OLLAMA_PRIMARY_OPENAI = "http://localhost:11434/v1"
OLLAMA_ADVISOR_OPENAI = "http://127.0.0.1:11435/v1"
OLLAMA_PRIMARY_API = "http://localhost:11434"
OLLAMA_ADVISOR_API = "http://127.0.0.1:11435"

USE_ML_IN_GOLDEN = True        # чтобы можно было выкл. одной строкой
ML_AUTO_TRAIN_ENABLED = True
ML_AUTO_TRAIN_MIN_SAMPLES = 500
ML_AUTO_TRAIN_EPOCHS = 50
ML_TARGET_MIN_PNL_PCT = 0.5

ML_POLICY = {
    "ENABLED": True,
    "STATE_PATH": ROOT_DIR / "ml_policy_state.pkl",
    "MIN_TRAIN_SAMPLES": 500,
    "TARGET_PNL_PCT": 0.8,
    "ENTRY_THRESHOLD": 0.55,
    "EXIT_THRESHOLD": 0.35,
    "MIN_HOLD_SEC": 180.0,
    "SAVE_INTERVAL": 100,
    "MAX_BUFFER": 5000,
    "APPLY_SOURCES": ["golden"],
    "BYPASS_WHEN_UNFIT": True,
}

# вход по цене+объёму
GOLDEN_ENTER_ON_PRICE_VOLUME = True
GOLDEN_PV_PRICE_DELTA_1M_PCT = 0.8     # 0.8% по умолчанию, подбери под свой рынок
GOLDEN_PV_VOLUME_RATIO_1M    = 2.0     # x2 к предыдущей минуте

# жёсткие 3/3 (как раньше)
GOLDEN_REQ_PRICE_DELTA_1M_PCT = 0.30
GOLDEN_REQ_VOLUME_RATIO_1M    = 2.0
GOLDEN_REQ_DOI1M_PCT          = 0.35

# прочее
GOLDEN_ALLOW_DOI5M_IF_1M_MISSING = True
GOLDEN_PRERUN_CAP_PCT = 6.5
GOLDEN_ALERT_ON_2_OF_3 = True
GOLDEN_WATCH_DEBOUNCE_SEC = 30



# --- Ключи признаков для ML-модели ---
FEATURE_KEYS = [
    "price", "pct1m", "pct5m", "pct15m", "vol1m", "vol5m", "vol15m",
    "OI_now", "dOI1m", "dOI5m", "spread_pct", "sigma5m", "CVD1m", "CVD5m",
    "rsi14", "sma50", "ema20", "atr14", "bb_width", "supertrend", "cci20",
    "macd", "macd_signal", "avgVol30m", "avgOI30m", "deltaCVD30m",
    "GS_pct4m", "GS_vol4m", "GS_dOI4m", "GS_cvd4m", "GS_supertrend", "GS_cooldown",
    "SQ_pct1m", "SQ_pct5m", "SQ_vol1m", "SQ_vol5m", "SQ_dOI1m", "SQ_spread_pct",
    "SQ_sigma5m", "SQ_liq10s", "SQ_power", "SQ_strength", "SQ_cooldown",
    "LIQ_cluster_val10s", "LIQ_cluster_count10s", "LIQ_direction", "LIQ_pct1m",
    "LIQ_pct5m", "LIQ_vol1m", "LIQ_vol5m", "LIQ_dOI1m", "LIQ_spread_pct",
    "LIQ_sigma5m", "LIQ_golden_flag", "LIQ_squeeze_flag", "LIQ_cooldown",
    "hour_of_day", "day_of_week", "month_of_year", "adx14", "open_interest",
    "volume_1m", "rsi14_prev", "adx14_prev", "volume_anomaly_prev"
]
INPUT_DIM = len(FEATURE_KEYS)

# DOM_SQUEEZE_STRATEGY = {
#     "ENABLED": True,
#     "BREAKOUT_TRADING_ENABLED": True,
#     "FADE_TRADING_ENABLED": False, 
#     "SQUEEZE_LOOKBACK_MS": 5000,
#     "SQUEEZE_THRESHOLD_PERCENT": 0.2,
#     "WALL_MULTIPLIER": 10,
#     "WALL_SCAN_DEPTH": 20,
#     "COOLDOWN_SECONDS": 300,
#     "AVERAGING_ENABLED": True,
#     "MAX_AVERAGING_ORDERS": 1,   
#     "ZSCORE_LEVELS_N": 20,
#     "ZSCORE_MIN": 2.0,
#     "TAPE_TOUCH_ENABLED": True,
#     "TAPE_TOUCH_REQUIRED_FOR_FADE": True,
#     "TAPE_TOUCH_REQUIRED_FOR_BREAKOUT": False,
#     "BREAKOUT_HUNT_WINDOW_SEC": 180,
#     "BREAKOUT_CONFIRMATION_SCORE": 80,
#     "SCORE_PRICE_CONFIRMED": 40,
#     "SCORE_FLOW_CONFIRMED": 40,
#     "SCORE_VOLUME_CONFIRMED": 20,
#     "WALL_CLUSTER_RATIO": 1.6,
#     "MIN_WALL_RATING": 1,
#     "Z_MIN_FOR_BREAKOUT": 1.5,

#     # РЕКОМЕНДАЦИЯ: Если стратегия DOM генерирует мало сигналов,
#     # попробуйте расширить "серую зону", уменьшив BREAKOUT_CONFIDENCE_MIN
#     # и увеличив FADE_CONFIDENCE_MAX.
#     "FADE_CONFIDENCE_MAX": 0.30,
#     "BREAKOUT_CONFIDENCE_MIN": 0.70,
#     "ENTRY_MODE": "retest",

#     "RETEST_SETTINGS": {
#         "WAIT_WINDOW_SEC": 20.0,
#         "BAND_TICKS": 5,
#         "BOUNCE_TICKS": 5,
#         "BOUNCE_CONFIRM_SEC": 15.0,
#         "MAX_SPREAD_TICKS": 10
#     },
# }

DOM_SQUEEZE_STRATEGY = {
    "ENABLED": True,
    "MAX_SCAN_LEVELS": 80,
    "MAX_DISTANCE_TICKS": 120,
    "MIN_WALL_ZSCORE": 1.5,

    # усиленные требования для доверия к стене
    "MIN_Z_HIGH": 3.0,
    "MIN_Z_MEDIUM": 2.1,
    "MIN_Z_BREAKOUT": 2.4,
    "MIN_Z_FADE": 1.8,
    "MIN_RATING_HIGH": 5,
    "MAX_DISTANCE_PCT": 0.18,
    "MIN_WALL_RATING": 2,
    "WALL_BASE_SIZE": 22000,

    # дополнительные фильтры
    "RETEST_REQUIRED": True,
    "RETEST_TICKS": 3,
    "RETEST_MAX_SECONDS": 12,
    "REQUIRED_HOLD_RATIO": 0.65,
    "BREACH_COOLDOWN_SEC": 300,
    "CONFIRMATION_MKT_DELTA": 0.0,
    "MIN_CVD_DELTA": 45.0,
    "MIN_VOL_RATIO": 0.8,
    "MAX_SPREAD_PCT": 0.18,
    "ML_BREAKOUT_THRESHOLD": 0.62,
    "ML_FADE_THRESHOLD": 0.45,
    "FADE_MIN_RATING": 40,
    "FADE_MIN_HOLD_RATIO": 0.72,
    "FADE_MIN_ML_HOLD": 0.50,
    "BREAKOUT_MIN_RATING": 20,
    "BREAKOUT_MIN_HOLD_RATIO": 0.60,
    "BREAKOUT_MIN_ML_SCORE": 0.55
}


BREAKOUT_HUNTER = {
    # окно короче, чтобы не висеть зря
    "WINDOW_SEC": 20,
    # если у тебя были другие пороги — не меняй; ниже — только то, что нужно для чек-листа
    "REQUIRED_SCORE": 70,

    # кулдауны на перезапуск по тому же символу
    "COOLDOWN_AFTER_FILLED_SEC": 35,
    "COOLDOWN_AFTER_CANCEL_SEC": 30,

    # предфильтр по силе тренда и по спреду
    "MIN_ADX": 18.0,
    "MAX_SPREAD_TICKS": 4,
    "PRICE_DELTA_THRESHOLD": 0.2,
    "EXTREME_VOLUME_SPIKE": 75.0,
    "SCORE_VOLUME_EXTREME": 10,
}

GOLDEN_LIQUIDITY_FILTER = {
    "MIN_TURNOVER_24H": 15_000_000.0,
    "MIN_AVG_VOL_1M": 3_500.0,
}

AGGRESSIVE_GOLDEN_SETUP = {
    "ENABLED": True,
    "VOLUME_FACTOR": 3.0,
    "PRICE_DELTA_1M": 0.9,
    "PRICE_DELTA_4M": 1.8,
    "MAX_CONTEXT_ABS": 12.0,
    "LOOKBACK_MINUTES": 120,
    "COOLDOWN_SEC": 300.0,
    "TICK_INTERVAL_SEC": 2.0,
    "MIN_ALIGN_PCT15M": 0.2,
    "MIN_VOLUME_ANOMALY": 1.5,
    "MAX_SPREAD_PCT": 0.05,
}

GOLDEN_FAIL_FAST = {
    "ENABLED": False,
    "WAIT_FOR_FILL_SEC": 5.0,
    "WINDOW_SEC": 30.0,
    "LOSS_PCT": 0.6,
    "MIN_POSITIVE_PCT": 0.2,
    # Позиция переводится в безубыток, только если ROI достиг положительного
    # порога breakeven_trigger_pct (в процентах). Защитный стоп выставляется с
    # учётом комиссионного буфера breakeven_buffer_pct.
    "BREAKEVEN_TRIGGER_PCT": 4.0,
    # Комиссионный буфер указывается в процентах от цены (0.02% ~= 0.0002 при плече 10x соответствует 0.2% от объёма).
    "BREAKEVEN_BUFFER_PCT": 0.02,
    "MAX_HOLD_SEC": 1800.0,
}

POSITION_SECURITY = {
    "ENABLED": True,
    "PROFIT_THRESHOLD_PCT": 0.6,
    "LOCK_PERCENT_OF_PROFIT": 0.4,
    "LOCK_CAP_PCT": 2.5,
    "TRAIL_BUFFER_PCT": 0.25,
    "LOSS_HARD_STOP_PCT": 0.8,
    "AUTO_CLOSE_IF_LOSS": True,
    "RECHECK_INTERVAL_SEC": 900,
    "LOSS_GRACE_PERIOD_SEC": 900,
    "LOSS_PROTECT_RATING": 200,
    "LOSS_PROTECT_HOLD_RATIO": 0.75
}

# Максимально допустимая доля потерь от маржи на одну сделку (в процентах).
MAX_LEVERED_LOSS_PCT = 50.0


# Настройки трейлинга. НЕ ИЗМЕНЯЛИСЬ.
ACTIVE_TRAILING_MODE = "adaptive" # возможные режимы "simple_gap", "adaptive", "dynamic"
# TRAILING_MODES = {
#     "simple_gap": {
#         "ACTIVATION_PNL_PCT": 0.5,
#         "TRAILING_GAP_PCT": 0.30
#     },
#     "dynamic": {
#         "MIN_UPDATE_INTERVAL_SECS": 1.0,
#         "MIN_LOG_INTERVAL_SECS": 5.0,
#         "TRAIL_ACTIVATE_ROI_LEVERED_PCT": 5.0,
#         "ASSUME_LEVERAGE_IF_MISSING": 10.0,
#         "PRE_WALL_ATR_K": 3.0,
#         "K_DEFAULT": 2.5,
#         "K_MIN": 0.8,
#         "ROI_TIERS": [
#             {"roi": 0.0,  "band": "adx_lt_20", "k": 2.2}, {"roi": 0.0,  "band": "adx_ge_20", "k": 2.5}, {"roi": 0.0,  "band": "adx_gt_30", "k": 2.8},
#             {"roi": 0.5,  "band": "adx_lt_20", "k": 2.0}, {"roi": 0.5,  "band": "adx_ge_20", "k": 2.3}, {"roi": 0.5,  "band": "adx_gt_30", "k": 2.6},
#             {"roi": 1.0,  "band": "adx_lt_20", "k": 1.8}, {"roi": 1.0,  "band": "adx_ge_20", "k": 2.1}, {"roi": 1.0,  "band": "adx_gt_30", "k": 2.4}
#         ],
#         "FLOW_TIGHTEN_FACTOR": 0.9,
#         "BREAKEVEN_BUFFER_PCT": 0.18,
#         "BREAKEVEN_ARM_SEC": 20.0,
#         "BREAKEVEN_ARM_ROI_PCT": 0.25,
#         "MIN_GAP_TICKS": 2,
#         "URGENCY_EXTRA_GAP_TICKS": 1,
#         "ADX_SLOPE_WINDOW": 6,
#         "FADE_ADX_LT": 18.0,
#         "FADE_ADX_SLOPE_DOWN_PM": 4.0,
#         "FADE_WEIGHT_LOW_ADX": 0.5,
#         "FADE_WEIGHT_SLOPE": 0.5,
#         "WALL_BAND_TICKS": 15,
#         "WALL_STALL_SEC": 5.0,
#         "WALL_STALL_MAX_EXTRA": 10.0,
#         "FADE_URGENCY_WEIGHT": 0.6,
#         "WALL_URGENCY_WEIGHT": 0.6,
#         "MAX_TIGHTEN_URGENCY": 0.9,
#         "URGENCY_K_SHRINK": 0.6
#     }
# }


TRAILING_MODES = {
    "simple_gap": {
        "ACTIVATION_PNL_PCT": 0.5,
        "TRAILING_GAP_PCT": 0.30,
        "LOG_THROTTLE_SEC": 2.0,
        "MIN_LOG_DELTA_TICKS": 2,
    },

    "dynamic": {
        # активация трейлинга по ROI С УЧЁТОМ ПЛЕЧА:
        "TRAIL_ACTIVATE_ROI_LEVERED_PCT": 5.0,   # напр. 5.0 → ROI@x10 достигается при ~0.5% хода цены
        "ASSUME_LEVERAGE_IF_MISSING": 10.0,      # дефолтное плечо, если не нашли в позиции

        # PRE-фаза: стоп от противоположной стены с запасом
        "PRE_WALL_ATR_K": 3.0,

        # тайминги
        "MIN_UPDATE_INTERVAL_SECS": 0.3,
        "MIN_LOG_INTERVAL_SECS": 12.0,

        # биржевые ограничения
        "MIN_GAP_TICKS": 2,
        "PEAK_LOCK_TICKS_MIN": 2,
        "PEAK_LOCK_ATR_MULT": 0.45,
        "FADE_TO_PEAK_WEIGHT": 1.0,

        # динамика «срочности» при застое у стены
        "URGENCY_EXTRA_GAP_TICKS": 1,
        "FADE_URGENCY_WEIGHT": 0.6,
        "WALL_URGENCY_WEIGHT": 0.6,
        "MAX_TIGHTEN_URGENCY": 0.9,
        "URGENCY_K_SHRINK": 0.6,

        # ускорение, когда тренд утомляется
        "SLOWDOWN_ADX_SLOPE_PM": 6.0,
        "SLOWDOWN_ROI_DECAY_SEC": 20.0,
        "SLOWDOWN_ACCEL_WEIGHT": 0.65,
        "SLOWDOWN_MIN_GAP_TICKS": 1,
        "STALL_TIGHTEN_AFTER_SEC": 25.0,
        "STALL_TIGHTEN_STEP_SEC": 10.0,
        "STALL_TIGHTEN_STEP_PCT": 0.12,
        "STALL_TIGHTEN_MAX": 0.60,

        # ADX / ROI → коэффициент k
        "K_DEFAULT": 2.5,
        "K_MIN": 0.8,
        "FLOW_TIGHTEN_FACTOR": 0.85,
        "ROI_TIERS": [
            {"roi": 0.3, "band": "adx_lt_20", "k": 2.5},
            {"roi": 0.3, "band": "adx_ge_20", "k": 2.7},
            {"roi": 0.3, "band": "adx_gt_30", "k": 3.0},
            {"roi": 0.8, "band": "adx_lt_20", "k": 2.2},
            {"roi": 0.8, "band": "adx_ge_20", "k": 2.4},
            {"roi": 0.8, "band": "adx_gt_30", "k": 2.6},
            {"roi": 1.5, "band": "adx_lt_20", "k": 2.0},
            {"roi": 1.5, "band": "adx_ge_20", "k": 2.2},
            {"roi": 1.5, "band": "adx_gt_30", "k": 2.4},
        ],

        # ADX fade
        "ADX_SLOPE_WINDOW": 6,
        "FADE_ADX_LT": 18.0,
        "FADE_ADX_SLOPE_DOWN_PM": 4.0,
        "FADE_WEIGHT_LOW_ADX": 0.5,
        "FADE_WEIGHT_SLOPE": 0.5,

        # «застревание» у стены
        "WALL_BAND_TICKS": 4,
        "WALL_STALL_SEC": 8.0,
        "WALL_STALL_MAX_EXTRA": 10.0,

        # безубыток (армирование — по цене, без плеча)
        "BREAKEVEN_BUFFER_PCT": 0.18,
        "BREAKEVEN_ARM_SEC": 20.0,
        "BREAKEVEN_ARM_ROI_PCT": 0.25,
    }
}


TRAILING_PROFILES = {
    # Мажоры — ход чище, вола предсказуемее
    "major": {
        "symbols": ["BTCUSDT", "ETHUSDT", "BTCUSD", "ETHUSD"],
        "ATR_LEN": 14,
        "K": {  # множители ATR для TB по режимам
            "AGGR_START": (0.70, 1.20),
            "SLOW_TREND": (1.30, 2.00),
            "RANGE":      (2.20, 3.20),
        },
        "TB_MIN": 0.60, "TB_MAX": 3.20,           # ограничения TB в ATR
        "ROI_ACTIVATION": 0.04,                   # включать трейлинг с 4% ROI
        "COOLDOWN_SEC": 0.80,                     # кулдаун обновления стопа
        "SPIKY_TAIL_RATIO": 0.66,                 # фильтр «игл»
        "SPIKE_TIGHTEN_MULT": 0.55,
        "SPIKE_FORCE_TICKS": 2,
        "SPIKE_VOL_THRESHOLD": 4.5,
        "SPIKE_PRICE_THRESHOLD": 0.9,
        "BREAKEVEN_BUFFER_PCT": 0.12,
        "BREAKEVEN_ARM_SEC": 12.0,
        "BREAKEVEN_ARM_ROI_PCT": 0.30,
        "STALL_TIGHTEN_AFTER_SEC": 25.0,
        "STALL_TIGHTEN_STEP_SEC": 10.0,
        "STALL_TIGHTEN_STEP_PCT": 0.12,
        "STALL_TIGHTEN_MAX": 0.60,
    },

    # Высокобетовые альты — шумнее, глубже откаты
    "alt": {
        "symbols": [],  # добавь сюда свои любимые альты
        "ATR_LEN": 14,
        "K": {
            "AGGR_START": (0.90, 1.60),
            "SLOW_TREND": (1.60, 2.40),
            "RANGE":      (2.60, 3.50),
        },
        "TB_MIN": 0.70, "TB_MAX": 3.50,
        "ROI_ACTIVATION": 0.06,
        "COOLDOWN_SEC": 1.20,
        "SPIKY_TAIL_RATIO": 0.66,
        "SPIKE_TIGHTEN_MULT": 0.50,
        "SPIKE_FORCE_TICKS": 3,
        "SPIKE_VOL_THRESHOLD": 5.0,
        "SPIKE_PRICE_THRESHOLD": 1.1,
        "BREAKEVEN_BUFFER_PCT": 0.12,
        "BREAKEVEN_ARM_SEC": 12.0,
        "BREAKEVEN_ARM_ROI_PCT": 0.30,
        "STALL_TIGHTEN_AFTER_SEC": 30.0,
        "STALL_TIGHTEN_STEP_SEC": 12.0,
        "STALL_TIGHTEN_STEP_PCT": 0.14,
        "STALL_TIGHTEN_MAX": 0.65,
    },

    # Илликвид/новички — ещё шире, чтобы не выбивало
    "illiquid": {
        "symbols": [],  # укажи символы, где частые «шипы»
        "ATR_LEN": 14,
        "K": {
            "AGGR_START": (1.10, 1.80),
            "SLOW_TREND": (1.90, 2.80),
            "RANGE":      (3.00, 4.20),
        },
        "TB_MIN": 0.90, "TB_MAX": 4.20,
        "ROI_ACTIVATION": 0.06,
        "COOLDOWN_SEC": 1.50,
        "SPIKY_TAIL_RATIO": 0.66,
        "SPIKE_TIGHTEN_MULT": 0.45,
        "SPIKE_FORCE_TICKS": 3,
        "SPIKE_VOL_THRESHOLD": 5.5,
        "SPIKE_PRICE_THRESHOLD": 1.3,
        "BREAKEVEN_BUFFER_PCT": 0.12,
        "BREAKEVEN_ARM_SEC": 12.0,
        "BREAKEVEN_ARM_ROI_PCT": 0.30,
        "STALL_TIGHTEN_AFTER_SEC": 35.0,
        "STALL_TIGHTEN_STEP_SEC": 15.0,
        "STALL_TIGHTEN_STEP_PCT": 0.16,
        "STALL_TIGHTEN_MAX": 0.68,
    },
}

INTERSESSION_TRADING_ENABLED = False
WEEKEND_INTERSESSION_ENABLED = False
WEEKEND_INTERSESSION_DAYS = {5, 6}  # 5=Saturday, 6=Sunday


# Межсессионный скан: как часто пробовать вход по символу (если нет позиции)
INTERSESSION_THROTTLE_SEC = 15.0

INTERSESSION_CONFIG = {
    "LIQUIDITY": {
        "SPREAD_PCT_Q": 0.75,          # допустимый квантиль спреда
        "DEPTH_RATIO_MIN": 0.35,       # объём bid/ask в первых 10 тиках относительно дневной медианы
        "ORDERFLOW_IMBALANCE_MAX": 0.55,
        "TURNOVER_MIN_USDT": 8_000_000.0,
    },
    "VOLATILITY": {
        "OVERNIGHT_ATR_Q": 0.70,
        "RANGE_Z_MAX": 1.8,
        "JUMP_SIGMA": 3.0,
    },
    "CORRELATION": {
        "BETA_BTC_MAX": 0.65,
        "BETA_ETH_MAX": 0.55,
        "DIVERGENCE_PCT_MAX": 0.45,
    },
    "RISK": {
        "VAR_LIMIT_USDT": 120.0,
        "MAX_HOLD_MINUTES": 120,
        "MAX_OPEN_POSITIONS": 5,
    },
    "THROTTLE": {
        "ADAPTIVE": True,
        "BASE_SEC": INTERSESSION_THROTTLE_SEC,
        "SPREAD_PENALTY": 2.5,
        "DEPTH_BONUS": 0.6,
        "MIN_SEC": 10.0,
        "MAX_SEC": 45.0,
    },
}


# Профиль по умолчанию, если символ не попал ни в один класс
DEFAULT_TRAILING_PROFILE = {
    "ATR_LEN": 14,
    "K": {
        "AGGR_START": (0.80, 1.40),
        "SLOW_TREND": (1.50, 2.30),
        "RANGE":      (2.20, 3.20),
    },
    "TB_MIN": 0.70, "TB_MAX": 3.20,
    "ROI_ACTIVATION": 0.05,
    "COOLDOWN_SEC": 1.00,
    "SPIKY_TAIL_RATIO": 0.66,
    "SPIKE_TIGHTEN_MULT": 0.55,
    "SPIKE_FORCE_TICKS": 2,
    "SPIKE_VOL_THRESHOLD": 4.0,
    "SPIKE_PRICE_THRESHOLD": 1.0,
    "BREAKEVEN_BUFFER_PCT": 0.12,
    "BREAKEVEN_ARM_SEC": 12.0,
    "BREAKEVEN_ARM_ROI_PCT": 0.30,
    "STALL_TIGHTEN_AFTER_SEC": 28.0,
    "STALL_TIGHTEN_STEP_SEC": 12.0,
    "STALL_TIGHTEN_STEP_PCT": 0.13,
    "STALL_TIGHTEN_MAX": 0.62,
}

# Общие флаги
TRAILING_COOLDOWN_SEC = 1.0          # фолбэк, если профиль не задан
TRAILING_ACTIVATION_ROI = 0.05       # фолбэк
TRAILING_SPIKE_TIGHTEN_MULT = 0.55
TRAILING_SPIKE_FORCE_TICKS = 2
TRAILING_SPIKE_VOL_THRESHOLD = 4.0
TRAILING_SPIKE_PRICE_THRESHOLD = 1.0

# Минимальная удаленность стартового стопа от цены входа (в % по рынку).
MIN_INITIAL_STOP_PCT = 3.0




FLEA_STRATEGY = {
    "ENABLED": True,
    "MAX_OPEN_POSITIONS": 15,
    "FAST_EMA_PERIOD": 5, "SLOW_EMA_PERIOD": 10, "TREND_EMA_PERIOD": 200,
    "RSI_CONFIRMATION_PERIOD": 14, "ATR_PERIOD": 14,
    "MIN_ATR_PCT": 0.05, "MAX_ATR_PCT": 1.5,
    "TP_ATR_MULTIPLIER": 1.5, "STOP_LOSS_ATR_MULTIPLIER": 1.0,
    "MAX_HOLD_MINUTES": 10
}

# --- ИСПРАВЛЕННАЯ ВЕРСИЯ ---
# Упрощены настройки, так как проверка на retrace была убрана из кода,
# а REQUIRE_...ALIGNMENT флаги не использовались в исправленной версии.
ENTRY_GUARD = {
    "MAX_SPREAD_PCT": 0.25,
    # Блок по импульсу ПРОТИВ контртрендовой сделки
    "PUMP_BLOCK_1M_PCT": 1.2,
    "PUMP_BLOCK_5M_PCT": 3.0,
    "DUMP_BLOCK_1M_PCT": 1.2,
    "DUMP_BLOCK_5M_PCT": 3.0,
    # Кулдаун после блокировки по импульсу (сек)
    "MOMENTUM_COOLDOWN_SEC": 90,
}

TRADES_UNIFIED_BASE_HEADERS = [
    "timestamp",
    "symbol",
    "side",
    "event",
    "result",
    "volume_trade",
    "price_trade",
    "pnl_usdt",
    "pnl_pct",
    "price",
    "open_interest",
    "volume_1m",
    "rsi14",
    "adx14",
    "volume_anomaly",
    "source",
]

# trades_unified.csv теперь хранит полный вектор признаков, используемых ML.
# Это позволяет переобучать модель на тех же данных, что и онлайн-инференс.
TRADES_UNIFIED_CSV_HEADERS = TRADES_UNIFIED_BASE_HEADERS + FEATURE_KEYS

def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    rotating_handler = RotatingFileHandler(LOG_FILE, maxBytes=50*1024*1024, backupCount=5)
    rotating_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger.addHandler(rotating_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger.addHandler(console_handler)

    wallet_logger = logging.getLogger("wallet_state")
    wallet_logger.setLevel(logging.INFO)
    wallet_handler = RotatingFileHandler(WALLET_LOG_FILE, maxBytes=20 * 1024 * 1024, backupCount=3)
    wallet_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    wallet_logger.addHandler(wallet_handler)
    wallet_logger.propagate = False

    logging.info("[ML] Using compute device: %s", BOT_DEVICE)
