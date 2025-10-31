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
WALL_MEMORY_FILE = Path(__file__).parent / "wall_memory.pkl" # <--- ДОБАВЬТЕ ЭТУ СТРОКУ
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
    "BREAKOUT_TRADING_ENABLED": True,   # РАЗРЕШИТЬ торговлю на пробой
    "FADE_TRADING_ENABLED": False,      # ЗАПРЕТИТЬ торговлю на отбой (пока)
    "SQUEEZE_LOOKBACK_MS": 5000,
    "SQUEEZE_THRESHOLD_PERCENT": 0.2,
    "WALL_MULTIPLIER": 10,
    "WALL_SCAN_DEPTH": 20,
    "COOLDOWN_SECONDS": 300,
    "AVERAGING_ENABLED": True,
    "AVERAGING_WALL_MULTIPLIER": 15.0,
    "AVERAGING_MIN_DISTANCE_PCT": 5.0,
    "MAX_AVERAGING_ORDERS": 1,   
    "ZSCORE_LEVELS_N": 20,
    "ZSCORE_MIN": 2.0,
    "WALL_PROXIMITY_TICKS": 5,
    "TAPE_TOUCH_ENABLED": True,
    "TAPE_TOUCH_REQUIRED_FOR_FADE": True,
    "TAPE_TOUCH_REQUIRED_FOR_BREAKOUT": False,
    "TAPE_TOUCH_LB_SECS": 3,
    "TAPE_TOUCH_MAX_DIST_TICKS": 3,
    "TAPE_TOUCH_MIN_TRADES": 5,
    "TAPE_TOUCH_MIN_ABSORB_QTY": 10000,
    "BREAKOUT_THRESHOLD_BASE": 0.62,
    "HOLD_THRESHOLD_BASE": 0.58,
    "MIN_PROB_DEVIATION": 0.05,
    "BREAKOUT_HUNT_WINDOW_SEC": 180,       # Уменьшаем окно до 3 минут
    "BREAKOUT_CONFIRMATION_SCORE": 80,     # Порог в баллах для входа
    "SCORE_PRICE_CONFIRMED": 40,           # Баллы за подтверждение ценой
    "SCORE_FLOW_CONFIRMED": 40,            # Баллы за подтверждение потоком (CVD)
    "SCORE_VOLUME_CONFIRMED": 20,          # Баллы за подтверждение объемом

    # --- НАЧАЛО НОВЫХ КЛЮЧЕВЫХ ПАРАМЕТРОВ ---
    # На основе вашего наблюдения: низкая уверенность = сигнал к развороту (fade)
    "FADE_CONFIDENCE_MAX": 0.30,
    # Гипотеза: высокая уверенность = сигнал к пробою (breakout)
    "BREAKOUT_CONFIDENCE_MIN": 0.70,
    "ENTRY_MODE": "retest",  # 'retest' или 'immediate'

    "RETEST_SETTINGS": {
        "WAIT_WINDOW_SEC": 20.0,
        "BAND_TICKS": 5,
        "BOUNCE_TICKS": 5,
        "BOUNCE_CONFIRM_SEC": 15.0,
        "MAX_SPREAD_TICKS": 10
    },
}


# TRAILING_SETTINGS = {
#     "MIN_UPDATE_INTERVAL_SECS": 1.6,
#     "MIN_STOP_TICKS": 3,
#     "BREAKEVEN_BUFFER_PCT": 0.18,  # комиссии+спред в %, точнее чем фикс 0.2
#     "ROI_TIERS": [
#     {"roi": 0.0,  "band": "adx_gt_30", "k": 3.2},
#     {"roi": 0.0,  "band": "adx_ge_20", "k": 2.6},
#     {"roi": 0.0,  "band": "adx_lt_20", "k": 2.0},
#     {"roi": 6.0,  "band": "adx_gt_30", "k": 2.6},
#     {"roi": 6.0,  "band": "adx_ge_20", "k": 2.2},
#     {"roi": 6.0,  "band": "adx_lt_20", "k": 1.8},
#     {"roi": 10.0, "band": "adx_gt_30", "k": 2.2},
#     {"roi": 10.0, "band": "adx_ge_20", "k": 1.8},
#     {"roi": 10.0, "band": "adx_lt_20", "k": 1.5}
#     ],
#     "FLOW_TIGHTEN_FACTOR": 0.85,
#     "SWING_LOOKBACK": 5,
#     "SWING_BUFFER_TICKS": 2
#     }

# --- НАЧАЛО КЛЮЧЕВОГО ИЗМЕНЕНИЯ: Новая структура настроек трейлинга ---
# Выберите активный режим: "dynamic" или "simple_gap"
ACTIVE_TRAILING_MODE = "dynamic"

TRAILING_MODES = {
    "simple_gap": {
        # Активируем трейлинг, когда позиция в плюсе не меньше этого процента
        "ACTIVATION_PNL_PCT": 0.5,
        # Фиксированный зазор стопа от цены (в процентах)
        "TRAILING_GAP_PCT": 0.30
    },

    "dynamic": {
        # Тайминги/лог
        "MIN_UPDATE_INTERVAL_SECS": 0.5,   # минимум между пересчётами трейлинга
        "MIN_LOG_INTERVAL_SECS": 1.0,     # антиспам логов

        "TRAIL_ACTIVATE_ROI_LEVERED_PCT": 5.0,   # целевой ROI с плечом для активации динамического трейла
        "ASSUME_LEVERAGE_IF_MISSING": 10.0,      # если в позиции/картах нет плеча — принять x10
        "PRE_WALL_ATR_K": 5.0,                   # коэффициент «х3 от плотности» (через 3*ATR от стены)

        # Базовые коэффы ATR*k
        "K_DEFAULT": 2.5,
        "K_MIN": 0.8,

        # Таблица подбора k по ROI от ПИКА и диапазону ADX
        # band ∈ {"adx_lt_20","adx_ge_20","adx_gt_30"}
        "ROI_TIERS": [
            {"roi": 0.0,  "band": "adx_lt_20", "k": 2.2},
            {"roi": 0.0,  "band": "adx_ge_20", "k": 2.5},
            {"roi": 0.0,  "band": "adx_gt_30", "k": 2.8},

            {"roi": 0.5,  "band": "adx_lt_20", "k": 2.0},
            {"roi": 0.5,  "band": "adx_ge_20", "k": 2.3},
            {"roi": 0.5,  "band": "adx_gt_30", "k": 2.6},

            {"roi": 1.0,  "band": "adx_lt_20", "k": 1.8},
            {"roi": 1.0,  "band": "adx_ge_20", "k": 2.1},
            {"roi": 1.0,  "band": "adx_gt_30", "k": 2.4}
        ],

        # «Поток»: если идём ПРОТИВ потока (CVD5m против стороны), немного ужесточаем k
        "FLOW_TIGHTEN_FACTOR": 0.9,

        # Армирование безубытка
        "BREAKEVEN_BUFFER_PCT": 0.18,  # мин. буфер над/под входом в %
        "BREAKEVEN_ARM_SEC": 20.0,     # активировать безубыток не раньше, чем через N секунд
        "BREAKEVEN_ARM_ROI_PCT": 0.25, # или как только ROI позиции ≥ X %

        # Минимальные отступы стопа от текущей цены
        "MIN_GAP_TICKS": 2,            # базовый минимум
        "URGENCY_EXTRA_GAP_TICKS": 1,  # добавка к отступу пропорционально срочности u

        # ADX-затухание тренда (расчёт «fade»)
        "ADX_SLOPE_WINDOW": 6,             # сколько последних замеров ADX держим (шт.)
        "FADE_ADX_LT": 18.0,               # ADX ниже — считаем тренд слабым
        "FADE_ADX_SLOPE_DOWN_PM": 4.0,     # падение ADX (пунктов/мин) для «слабого» наклона
        "FADE_WEIGHT_LOW_ADX": 0.5,        # вклад слабого уровня ADX в fade
        "FADE_WEIGHT_SLOPE": 0.5,          # вклад негативного наклона ADX в fade

        # «Застревание» у DOM-стены
        "WALL_BAND_TICKS": 15,              # считаем у стены, если расстояние ≤ N тиков
        "WALL_STALL_SEC": 5.0,             # сколько секунд «жаться» у стены до начала ужесточения
        "WALL_STALL_MAX_EXTRA": 10.0,      # за доп. N секунд от 0 до 1.0 по шкале u_wall

        # Интегральная «срочность» ужесточения
        "FADE_URGENCY_WEIGHT": 0.6,        # вес fade(ADX) в общей срочности u
        "WALL_URGENCY_WEIGHT": 0.6,        # вес застревания у стены в u
        "MAX_TIGHTEN_URGENCY": 0.9,        # потолок u
        "URGENCY_K_SHRINK": 0.6            # насколько уменьшаем k при u=1.0 (k ← k*(1 - 0.6))
    }
}

# === BREAKOUT / HUNTER V3 ===
BREAKOUT_HUNTER = {
    "WINDOW_SEC": 300,               # окно «слежки» за уликaми
    "REQUIRED_SCORE": 68,            # порог для входа
    "EARLY_ENTRY_SCORE": 56,         # «скаут» — ранний неполный вход
    "ESCALATE_SCORE": 72,            # докрутка до полного лота
    "SCOUT_QTY_FRAC": 0.35,          # доля лота для «скаута»
    "SCORE_HEARTBEAT_SEC": 2.0,      # лог прогресса каждые 2с

    # AI согласие: авто-байпас при высокой уверенности
    "SKIP_AI_IF_SCORE_GE": 75,       # если score ≥ 75 — не спрашиваем AI
    "REQUIRE_AI_ELSE": True,         # иначе — спрашиваем как раньше

    # Кулдауны по символу
    "COOLDOWN_AFTER_FILLED_SEC": 15, # после сделки
    "COOLDOWN_AFTER_CANCEL_SEC": 8   # после истечения окна без входа
}

# === РИСК-ЛИМИТЫ ===
RISK_LIMITS = {
    "MAX_CONCURRENT_POSITIONS": 6,   # общий лимит одновременно открытых поз
    "MAX_PER_SYMBOL_POSITIONS": 1,   # one-way
    "MAX_OPEN_EXPOSURE_USDT": 6000,  # суммарная экспозиция, $
    "DAILY_TRADE_CAP": 60            # кап на число сделок в сутки
}


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
