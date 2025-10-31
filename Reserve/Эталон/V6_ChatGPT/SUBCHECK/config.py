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
AI_PRIMARY_MODEL = "trading-analyst"
AI_ADVISOR_MODEL = "0xroyce/plutus:latest"
OLLAMA_PRIMARY_OPENAI = "http://localhost:11434/v1"
OLLAMA_ADVISOR_OPENAI = "http://127.0.0.1:11435/v1"
OLLAMA_PRIMARY_API = "http://localhost:11434"
OLLAMA_ADVISOR_API = "http://127.0.0.1:11435"

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

DOM_SQUEEZE_STRATEGY = {
    "ENABLED": True,
    "BREAKOUT_TRADING_ENABLED": True,
    "FADE_TRADING_ENABLED": False, 
    "SQUEEZE_LOOKBACK_MS": 5000,
    "SQUEEZE_THRESHOLD_PERCENT": 0.2,
    "WALL_MULTIPLIER": 10,
    "WALL_SCAN_DEPTH": 20,
    "COOLDOWN_SECONDS": 300,
    "AVERAGING_ENABLED": True,
    "MAX_AVERAGING_ORDERS": 1,   
    "ZSCORE_LEVELS_N": 20,
    "ZSCORE_MIN": 2.0,
    "TAPE_TOUCH_ENABLED": True,
    "TAPE_TOUCH_REQUIRED_FOR_FADE": True,
    "TAPE_TOUCH_REQUIRED_FOR_BREAKOUT": False,
    "BREAKOUT_HUNT_WINDOW_SEC": 180,
    "BREAKOUT_CONFIRMATION_SCORE": 80,
    "SCORE_PRICE_CONFIRMED": 40,
    "SCORE_FLOW_CONFIRMED": 40,
    "SCORE_VOLUME_CONFIRMED": 20,

    # РЕКОМЕНДАЦИЯ: Если стратегия DOM генерирует мало сигналов,
    # попробуйте расширить "серую зону", уменьшив BREAKOUT_CONFIDENCE_MIN
    # и увеличив FADE_CONFIDENCE_MAX.
    "FADE_CONFIDENCE_MAX": 0.30,
    "BREAKOUT_CONFIDENCE_MIN": 0.70,
    "ENTRY_MODE": "retest",

    "RETEST_SETTINGS": {
        "WAIT_WINDOW_SEC": 20.0,
        "BAND_TICKS": 5,
        "BOUNCE_TICKS": 5,
        "BOUNCE_CONFIRM_SEC": 15.0,
        "MAX_SPREAD_TICKS": 10
    },
}

BREAKOUT_HUNTER = {
    # окно короче, чтобы не висеть зря
    "WINDOW_SEC": 120,
    # если у тебя были другие пороги — не меняй; ниже — только то, что нужно для чек-листа
    "REQUIRED_SCORE": 80,

    # кулдауны на перезапуск по тому же символу
    "COOLDOWN_AFTER_FILLED_SEC": 35,
    "COOLDOWN_AFTER_CANCEL_SEC": 30,

    # предфильтр по силе тренда и по спреду
    "MIN_ADX": 18.0,
    "MAX_SPREAD_TICKS": 4
}


# Настройки трейлинга. НЕ ИЗМЕНЯЛИСЬ.
ACTIVE_TRAILING_MODE = "dynamic"
TRAILING_MODES = {
    "simple_gap": {
        "ACTIVATION_PNL_PCT": 0.5,
        "TRAILING_GAP_PCT": 0.30
    },
    "dynamic": {
        "MIN_UPDATE_INTERVAL_SECS": 1.0,
        "MIN_LOG_INTERVAL_SECS": 5.0,
        "TRAIL_ACTIVATE_ROI_LEVERED_PCT": 5.0,
        "ASSUME_LEVERAGE_IF_MISSING": 10.0,
        "PRE_WALL_ATR_K": 3.0,
        "K_DEFAULT": 2.5,
        "K_MIN": 0.8,
        "ROI_TIERS": [
            {"roi": 0.0,  "band": "adx_lt_20", "k": 2.2}, {"roi": 0.0,  "band": "adx_ge_20", "k": 2.5}, {"roi": 0.0,  "band": "adx_gt_30", "k": 2.8},
            {"roi": 0.5,  "band": "adx_lt_20", "k": 2.0}, {"roi": 0.5,  "band": "adx_ge_20", "k": 2.3}, {"roi": 0.5,  "band": "adx_gt_30", "k": 2.6},
            {"roi": 1.0,  "band": "adx_lt_20", "k": 1.8}, {"roi": 1.0,  "band": "adx_ge_20", "k": 2.1}, {"roi": 1.0,  "band": "adx_gt_30", "k": 2.4}
        ],
        "FLOW_TIGHTEN_FACTOR": 0.9,
        "BREAKEVEN_BUFFER_PCT": 0.18,
        "BREAKEVEN_ARM_SEC": 20.0,
        "BREAKEVEN_ARM_ROI_PCT": 0.25,
        "MIN_GAP_TICKS": 2,
        "URGENCY_EXTRA_GAP_TICKS": 1,
        "ADX_SLOPE_WINDOW": 6,
        "FADE_ADX_LT": 18.0,
        "FADE_ADX_SLOPE_DOWN_PM": 4.0,
        "FADE_WEIGHT_LOW_ADX": 0.5,
        "FADE_WEIGHT_SLOPE": 0.5,
        "WALL_BAND_TICKS": 15,
        "WALL_STALL_SEC": 5.0,
        "WALL_STALL_MAX_EXTRA": 10.0,
        "FADE_URGENCY_WEIGHT": 0.6,
        "WALL_URGENCY_WEIGHT": 0.6,
        "MAX_TIGHTEN_URGENCY": 0.9,
        "URGENCY_K_SHRINK": 0.6
    }
}

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

TRADES_UNIFIED_CSV_HEADERS = [
    "timestamp", "symbol", "side", "event", "result",
    "volume_trade", "price_trade", "pnl_usdt", "pnl_pct", 
    "price", "open_interest", "volume_1m", "rsi14", "adx14",
    "volume_anomaly", "source"
]

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
