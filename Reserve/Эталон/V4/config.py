# config.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# --- Основные настройки ---
ADMIN_IDS = {36972091}  # Telegram ID администраторов
BOT_DEVICE = "cpu"      # Устройство для вычислений ML

# --- Пути к файлам ---
ROOT_DIR = Path(__file__).resolve().parent
LOG_FILE = ROOT_DIR / "bot.log"
WALLET_LOG_FILE = ROOT_DIR / "wallet_state.log"
HISTORY_FILE = ROOT_DIR / "history.pkl"
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

# --- НАЧАЛО ИЗМЕНЕНИЙ ---
DOM_SQUEEZE_STRATEGY = {
    "ENABLED": True,
    "SQUEEZE_LOOKBACK_MS": 5000,
    "SQUEEZE_THRESHOLD_PERCENT": 0.2,
    "WALL_MULTIPLIER": 10,
    "WALL_SCAN_DEPTH": 20,
    "WALL_PROXIMITY_TICKS": 5,
    "COOLDOWN_SECONDS": 300,
    # --- Новые параметры для усреднения по плотности ---
    "AVERAGING_ENABLED": True,                 # Включить/выключить логику усреднения
    "AVERAGING_WALL_MULTIPLIER": 15.0,         # Множитель для "стены" при усреднении (должен быть больше, чем для первого входа)
    "AVERAGING_MIN_DISTANCE_PCT": 5.0,         # Минимальная просадка в % от первого входа, чтобы начать искать вторую стену
    "MAX_AVERAGING_ORDERS": 1                  # Максимальное количество усредняющих ордеров
}
# --- КОНЕЦ ИЗМЕНЕНИЙ ---



FLEA_STRATEGY = {
    "ENABLED": True,
    "MAX_OPEN_POSITIONS": 15,
    # Периоды EMA
    "FAST_EMA_PERIOD": 5,
    "SLOW_EMA_PERIOD": 10,
    "TREND_EMA_PERIOD": 200,
    # Периоды индикаторов
    "RSI_CONFIRMATION_PERIOD": 14,
    "ATR_PERIOD": 14,
    # Фильтры волатильности
    "MIN_ATR_PCT": 0.05,
    "MAX_ATR_PCT": 1.5,
    # Управление позицией
    "TP_ATR_MULTIPLIER": 1.5,
    "STOP_LOSS_ATR_MULTIPLIER": 1.0,
    "MAX_HOLD_MINUTES": 10
}

# --- ML и AI ---
ML_CONFIDENCE_FLOOR = 0.65


# --- Hunter / Entry Guard defaults ---
ENTRY_GUARD = {
    # Максимально допустимый спред (в %)
    "MAX_SPREAD_PCT": 0.25,

    # Блок по импульсу ПРОТИВ нашей стороны
    # Для шорта блокируем, если цена за 1м/5м выросла сильнее порога;
    # для лонга — если упала сильнее порога.
    "PUMP_BLOCK_1M_PCT": 1.2,
    "PUMP_BLOCK_5M_PCT": 3.0,
    "DUMP_BLOCK_1M_PCT": 1.2,
    "DUMP_BLOCK_5M_PCT": 3.0,

    # Требовать совпадение потока (по желанию): рост + dOI+, CVD+ для блокировки шорта
    "REQUIRE_CVD_ALIGNMENT": True,
    "REQUIRE_OI_ALIGNMENT": True,

    # Минимальный откат от экстремума (в %), чтобы не входить «на пике»
    "MIN_RETRACE_FROM_EXTREME_PCT": 0.4,

    # Кулдаун после блокировки по импульсу (сек)
    "MOMENTUM_COOLDOWN_SEC": 90,
}

# --- ДОБАВЬТЕ ЭТОТ СПИСОК ---
TRADES_UNIFIED_CSV_HEADERS = [
    "timestamp", "symbol", "side", "event", "result",
    "volume_trade", "price_trade", "pnl_usdt", "pnl_pct", 
    "price", "open_interest", "volume_1m", "rsi14", "adx14",
    "volume_anomaly", "source"
    ]


# --- Настройка логирования ---
def setup_logging():
    """Настраивает ротируемые логи для основного бота и кошелька."""
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
