# telegram_bot.py
import csv
import json
import pathlib
import logging
import math
import re
import time
from datetime import datetime, timedelta

from aiogram import F, Router, types, Bot, Dispatcher
from aiogram.filters.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, ReplyKeyboardMarkup
from aiogram.filters import Command
from aiogram.enums import ParseMode

# Импортируем наши модули
import config
import utils

logger = logging.getLogger(__name__)

# --- Инициализация Telegram ---
# !!! ВАЖНО: ЗАМЕНИТЕ ЭТОТ ТОКЕН НА ВАШ !!!
TELEGRAM_BOT_TOKEN = "5833594543:AAE8BXe2G3tmvZTixAHGhGKzJIPN66P64a4" 

try:
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()
    router = Router()
    router_admin = Router()
    # Подключаем роутеры к диспетчеру
    dp.include_router(router)
    dp.include_router(router_admin)
except Exception as e:
    logging.critical(f"Не удалось инициализировать Telegram Bot: {e}. Проверьте токен.")
    bot = None
    dp = None
    router = None
    router_admin = None

# Глобальный реестр для доступа к ботам из обработчиков
GLOBAL_BOTS: list = []

# --- Вспомогательные функции для работы с JSON ---

def load_user_state() -> dict:
    if config.USER_STATE_FILE.exists():
        with config.USER_STATE_FILE.open("r", encoding="utf-8") as fp:
            try:
                return json.load(fp)
            except json.JSONDecodeError:
                return {}
    return {}

def save_user_state(data: dict) -> None:
    with config.USER_STATE_FILE.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)

def _load_json(path: pathlib.Path):
    if path.exists():
        with path.open("r", encoding="utf-8") as fp:
            try:
                return json.load(fp)
            except json.JSONDecodeError:
                return None
    return None

def _user_slice(all_data: dict, user_id: int | str, default=None):
    if all_data is None:
        return default
    return all_data.get(str(user_id), default)

# --- Функции для форматирования информационных сообщений ---

def _load_open_positions_from_csv() -> list[dict]:
    path = getattr(config, "TRADES_UNIFIED_CSV_PATH", config.ROOT_DIR / "trades_unified.csv")
    if not path.exists():
        return []
    positions: dict[str, dict] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                symbol = row.get("symbol")
                event = str(row.get("event", "")).lower()
                if not symbol or event not in {"open", "close"}:
                    continue
                if event == "open":
                    positions[symbol] = row
                else:
                    positions.pop(symbol, None)
    except Exception as exc:
        logger.error(f"Не удалось разобрать trades_unified.csv: {exc}", exc_info=True)
        return []

    # привести к списку и упорядочить по времени открытия
    def _parse_ts(value: str) -> float:
        try:
            return datetime.fromisoformat(value.replace("Z", "")).timestamp()
        except Exception:
            return 0.0

    result = []
    for symbol, row in positions.items():
        row = dict(row)
        row.setdefault("symbol", symbol)
        row["_ts"] = _parse_ts(row.get("timestamp", ""))
        result.append(row)
    result.sort(key=lambda r: r.get("_ts", 0.0))
    return result


def get_positions_summary(user_id: int) -> str:
    all_positions = _load_json(config.OPEN_POS_JSON)
    user_positions = _user_slice(all_positions, user_id, [])

    if not user_positions:
        user_positions = _load_open_positions_from_csv()

    if not user_positions:
        return "Нет открытых позиций."

    lines = ["🪙 <b>Открытые позиции:</b>"]
    for pos in user_positions:
        symbol = pos.get("symbol") or pos.get("Symbol") or "N/A"
        side = (pos.get("side") or pos.get("Side") or "").upper()
        qty = utils.safe_to_float(pos.get("volume") or pos.get("volume_trade") or pos.get("size") or 0.0)
        entry_price = utils.safe_to_float(pos.get("avg_price") or pos.get("price_trade") or pos.get("price") or 0.0)
        current_price = utils.safe_to_float(pos.get("current_price") or pos.get("last_price") or 0.0)
        stop_price = utils.safe_to_float(pos.get("last_stop_price") or pos.get("stopPrice") or 0.0)
        pnl_pct = utils.safe_to_float(pos.get("pnl_pct_est") or pos.get("pnl_pct") or 0.0)
        opened_at = pos.get("opened_at") or pos.get("timestamp") or pos.get("open_time")
        flags = []
        if pos.get("manual_mode"):
            flags.append("manual")
        if pos.get("adopted"):
            flags.append("adopted")

        parts = [f"• <code>{symbol}</code> {side or '—'}"]
        if qty:
            parts.append(f"объём {qty:.3f}")
        if entry_price:
            parts.append(f"вход {entry_price:.6f}")
        if current_price:
            parts.append(f"тек {current_price:.6f}")
        if stop_price:
            parts.append(f"стоп {stop_price:.6f}")
        if math.isfinite(pnl_pct) and pnl_pct:
            parts.append(f"PnL {pnl_pct:+.2f}%")
        if opened_at:
            parts.append(str(opened_at))
        if flags:
            parts.append(", ".join(flags))

        lines.append(" | ".join(parts))

    return "\n".join(lines)

def _load_wallet_from_log() -> dict | None:
    path = getattr(config, "WALLET_LOG_FILE", config.ROOT_DIR / "wallet_state.log")
    if not path.exists() or path.stat().st_size == 0:
        return None

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.error(f"Не удалось прочитать wallet_state.log: {exc}", exc_info=True)
        return None

    pattern = re.compile(r"IM\s*=\s*(?P<im>[-+]?\d+(?:\.\d+)?)")
    for line in reversed(lines):
        match = pattern.search(line)
        if match:
            return {"initial_margin": utils.safe_to_float(match.group("im"))}
    return None


def get_wallet_summary(user_id: int) -> str:
    all_wallets = _load_json(config.WALLET_JSON)
    wallet = _user_slice(all_wallets, user_id, {})
    if wallet:
        bal = wallet.get("totalEquity")
        avail = wallet.get("availableBalance")
        used = wallet.get("usedMargin")
        updated = wallet.get("timestamp")

        def _fmt(val):
            try:
                num = float(val)
                if math.isfinite(num):
                    return f"{num:.2f}"
            except (TypeError, ValueError):
                pass
            return str(val) if val not in (None, "") else "—"

        lines = ["💰 <b>Состояние кошелька</b>"]
        lines.append(f"Equity: <code>{_fmt(bal)}</code>")
        lines.append(f"Available: <code>{_fmt(avail)}</code>")
        lines.append(f"Used margin: <code>{_fmt(used)}</code>")
        if updated:
            lines.append(f"Обновлено: {updated}")
        return "\n".join(lines)

    snapshot = _load_wallet_from_log()
    if snapshot:
        im = snapshot.get("initial_margin")
        return (
            "💰 <b>Состояние кошелька</b>\n"
            f"Initial Margin: <code>{im:.2f} USDT</code>"
        )

    return "Нет данных кошелька."

def load_user_config(user_id: int) -> dict:
    return load_user_state().get(str(user_id), {}).copy()


def format_settings_snapshot(user_id: int) -> str:
    cfg = load_user_config(user_id)
    if not cfg:
        return "Настройки не найдены. Используются значения по умолчанию."

    volume = utils.safe_to_float(cfg.get("volume", 0.0))
    max_total = utils.safe_to_float(cfg.get("max_total_volume", 0.0))
    strategy_mode = cfg.get("strategy_mode", "full")
    trailing_mode = cfg.get("trailing_mode") or getattr(config, "ACTIVE_TRAILING_MODE", "adaptive")
    trading_mode = cfg.get("mode", "demo").upper()
    bot_active = cfg.get("bot_active", True)
    sleep_mode = cfg.get("sleep_mode", False)

    trailing_start = cfg.get("trailing_start_pct") or {}
    trailing_gap = cfg.get("trailing_gap_pct") or {}
    modes_to_show = sorted(set([strategy_mode, *trailing_start.keys(), *trailing_gap.keys()]))

    strict_defaults = getattr(config, "ML_STRICT_FILTERS", {})
    user_strict = cfg.get("ML_STRICT_FILTERS") or {}
    strict_enabled = bool(user_strict.get("ENABLED", strict_defaults.get("ENABLED", True)))
    strict_min = utils.safe_to_float(user_strict.get("MIN_WORKING_ML", strict_defaults.get("MIN_WORKING_ML", 0.33)))
    strict_max = utils.safe_to_float(user_strict.get("MAX_WORKING_ML", strict_defaults.get("MAX_WORKING_ML", 0.85)))

    lines = [
        "⚙️ <b>Текущие настройки</b>",
        f"Режим запуска: <code>{trading_mode}</code>",
        f"Стратегия: <code>{strategy_mode}</code>",
        f"Режим трейлинга: <code>{trailing_mode}</code>",
        f"Лимит позиции: <b>{volume:.0f} USDT</b>",
        f"Лимит портфеля: <b>{max_total:.0f} USDT</b>",
        f"Состояние бота: <b>{'Запущен' if bot_active else 'Остановлен'}</b>",
        f"Режим тишины: <b>{'ВКЛ' if sleep_mode else 'ВЫКЛ'}</b>",
        f"ML фильтры: <b>{'ВКЛ' if strict_enabled else 'ВЫКЛ'}</b> (рабочий диапазон {strict_min:.2f}–{strict_max:.2f})",
    ]

    if modes_to_show:
        lines.append("\n🎯 <b>Профили трейлинга</b>")
        for mode in modes_to_show:
            start_val = trailing_start.get(mode)
            gap_val = trailing_gap.get(mode)
            parts = []
            if start_val is not None:
                parts.append(f"старт {start_val:.2f}%")
            if gap_val is not None:
                parts.append(f"gap {gap_val:.2f}%")
            detail = ", ".join(parts) if parts else "нет данных"
            lines.append(f"• {mode}: {detail}")

    return "\n".join(lines)


def format_pnl_change(hours: int) -> str:
    path = getattr(config, "TRADES_UNIFIED_CSV_PATH", config.ROOT_DIR / "trades_unified.csv")
    if not path.exists():
        return "Нет данных о сделках."

    cutoff = datetime.utcnow() - timedelta(hours=hours)
    total_pnl = 0.0
    trades = 0
    wins = 0
    best = None
    worst = None

    try:
        with path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                if str(row.get("event", "")).lower() != "close":
                    continue
                ts = row.get("timestamp")
                if not ts:
                    continue
                try:
                    trade_dt = datetime.fromisoformat(ts.replace("Z", ""))
                except ValueError:
                    continue
                if trade_dt < cutoff:
                    continue
                pnl = utils.safe_to_float(row.get("pnl_usdt"))
                if not math.isfinite(pnl):
                    continue
                trades += 1
                total_pnl += pnl
                if pnl > 0:
                    wins += 1
                if best is None or pnl > best:
                    best = pnl
                if worst is None or pnl < worst:
                    worst = pnl
    except Exception as exc:
        logger.error(f"Не удалось прочитать trades_unified.csv: {exc}", exc_info=True)
        return "Ошибка чтения журнала сделок."

    if trades == 0:
        return "Нет сделок за выбранный период."

    avg_pnl = total_pnl / trades
    win_rate = (wins / trades) * 100 if trades else 0.0
    period_start = cutoff.strftime("%d.%m %H:%M")
    period_end = datetime.utcnow().strftime("%d.%m %H:%M")

    summary = [
        f"Период: <code>{period_start} — {period_end} UTC</code>",
        f"Сделок: <b>{trades}</b>",
        f"PnL: <b>{total_pnl:+.2f} USDT</b>",
        f"Win rate: <b>{win_rate:.1f}%</b>",
        f"Средний результат: <b>{avg_pnl:+.2f} USDT</b>",
    ]
    if best is not None and worst is not None:
        summary.append(f"Лучший / худший: <code>{best:+.2f} / {worst:+.2f}</code>")
    return "\n".join(summary)

# --- Клавиатуры ---

STRATEGY_MODES = [
    "full",
    "golden_squeeze",
    "golden_only",
    "liquidation_only",
    "squeeze_only",
    "dom_squeeze_only",
]

TRAILING_MODES_AVAILABLE = ["adaptive", "dynamic", "simple_gap"]


def get_main_menu_keyboard() -> ReplyKeyboardMarkup:
    kb = [
        [types.KeyboardButton(text="📊 Статус"), types.KeyboardButton(text="📈 Отчёты")],
        [types.KeyboardButton(text="🛠 Торговые настройки")],
        [types.KeyboardButton(text="⚙️ Управление ботом"), types.KeyboardButton(text="🆘 Сервис")],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)


def get_status_menu_keyboard() -> ReplyKeyboardMarkup:
    kb = [
        [types.KeyboardButton(text="Открытые позиции"), types.KeyboardButton(text="Данные по кошельку")],
        [types.KeyboardButton(text="Активные настройки")],
        [types.KeyboardButton(text="⬅️ Назад")],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)


def get_settings_menu_keyboard() -> ReplyKeyboardMarkup:
    kb = [
        [types.KeyboardButton(text="Общий лимит"), types.KeyboardButton(text="Лимит позиции")],
        [types.KeyboardButton(text="Стратегия торговли"), types.KeyboardButton(text="Режим запуска")],
        [types.KeyboardButton(text="Режим трейлинга")],
        [types.KeyboardButton(text="Трейлинг: Порог"), types.KeyboardButton(text="Трейлинг: Отступ")],
        [types.KeyboardButton(text="API ключи")],
        [types.KeyboardButton(text="⬅️ Назад")],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)


def get_trailing_logic_keyboard(user_id: int) -> ReplyKeyboardMarkup:
    cfg = load_user_config(user_id)
    stored_modes = set((cfg.get("trailing_start_pct") or {}).keys()) | set((cfg.get("trailing_gap_pct") or {}).keys())
    rows = [[types.KeyboardButton(text=mode)] for mode in sorted(set([*stored_modes, *STRATEGY_MODES]))]
    rows.append([types.KeyboardButton(text="Текущая стратегия")])
    rows.append([types.KeyboardButton(text="⬅️ Назад")])
    return ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True)


def get_bot_control_keyboard() -> ReplyKeyboardMarkup:
    kb = [
        [types.KeyboardButton(text="Режим тишины"), types.KeyboardButton(text="ML фильтры")],
        [types.KeyboardButton(text="Остановить бот"), types.KeyboardButton(text="Запустить бот")],
        [types.KeyboardButton(text="⬅️ Назад")],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)


def get_reports_menu_keyboard() -> ReplyKeyboardMarkup:
    kb = [
        [types.KeyboardButton(text="PnL за 24ч"), types.KeyboardButton(text="PnL за 7д")],
        [types.KeyboardButton(text="PnL за 30д")],
        [types.KeyboardButton(text="⬅️ Назад")],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)


def get_service_menu_keyboard() -> ReplyKeyboardMarkup:
    kb = [
        [types.KeyboardButton(text="Обратная связь")],
        [types.KeyboardButton(text="⬅️ Назад")],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

# --- FSM Стейты ---

class UpdateAPIStates(StatesGroup):
    waiting_api_key = State()
    waiting_api_secret = State()

class MaxVolumeStates(StatesGroup):
    waiting_max_volume = State()

class PositionVolumeStates(StatesGroup):
    waiting_volume = State()

class FeedbackStates(StatesGroup):
    waiting_message = State()

class TrailingConfigStates(StatesGroup):
    waiting_logic = State()
    waiting_value = State()

# --- Обработчики команд и меню ---

@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("Главное меню", reply_markup=get_main_menu_keyboard())

# --- Навигация по главному меню ---
@router.message(F.text == "📊 Статус")
async def show_status_menu(message: Message):
    await message.answer("Раздел «Статус»", reply_markup=get_status_menu_keyboard())


@router.message(F.text == "🛠 Торговые настройки")
async def show_settings_menu(message: Message):
    await message.answer("Раздел «Торговые настройки»", reply_markup=get_settings_menu_keyboard())


@router.message(F.text == "⚙️ Управление ботом")
async def show_bot_control_menu(message: Message):
    await message.answer("Раздел «Управление ботом»", reply_markup=get_bot_control_keyboard())


@router.message(F.text == "📈 Отчёты")
async def show_reports_menu(message: Message):
    await message.answer("Раздел «Отчёты»", reply_markup=get_reports_menu_keyboard())


@router.message(F.text == "🆘 Сервис")
async def show_service_menu(message: Message):
    await message.answer("Раздел «Сервис»", reply_markup=get_service_menu_keyboard())


@router.message(F.text == "⬅️ Назад")
async def back_to_main(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Главное меню", reply_markup=get_main_menu_keyboard())

# --- Торговые настройки ---
@router.message(F.text == "API ключи")
async def trading_update_api(message: Message, state: FSMContext):
    await message.answer("Введите новый API_KEY:")
    await state.set_state(UpdateAPIStates.waiting_api_key)

@router.message(UpdateAPIStates.waiting_api_key)
async def trading_update_api_key(message: Message, state: FSMContext):
    await state.update_data(new_api_key=message.text.strip())
    await message.answer("Теперь введите новый API_SECRET:")
    await state.set_state(UpdateAPIStates.waiting_api_secret)

@router.message(UpdateAPIStates.waiting_api_secret)
async def trading_update_api_secret(message: Message, state: FSMContext):
    data = await state.get_data()
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})
    user_data["api_key"] = data["new_api_key"]
    user_data["api_secret"] = message.text.strip()
    all_state[user_id] = user_data
    save_user_state(all_state)
    await message.answer("✅ Ключи API обновлены.", reply_markup=get_settings_menu_keyboard())
    await state.clear()

@router.message(F.text == "Режим запуска")
async def trading_mode_menu(message: Message):
    """
    Показывает кнопки для выбора режима 'Боевой' или 'Демо'.
    """
    kb = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[
        [types.KeyboardButton(text="Боевой (Real)")],
        [types.KeyboardButton(text="Демо (Demo)")],
        [types.KeyboardButton(text="⬅️ Назад")],
    ])
    
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    current_mode = all_state.get(user_id, {}).get("mode", "real")
    
    await message.answer(
        f"Текущий режим: <b>{current_mode.upper()}</b>\n\n"
        "Выберите новый режим. <b>Внимание:</b> для вступления изменений в силу "
        "потребуется полный перезапуск бота.",
        reply_markup=kb,
        parse_mode=ParseMode.HTML
    )

@router.message(F.text.in_({"Боевой (Real)", "Демо (Demo)"}))
async def trading_set_mode(message: Message):
    """
    Сохраняет выбранный режим в user_state.json.
    """
    # Определяем режим на основе текста кнопки
    new_mode = "real" if "Real" in message.text else "demo"
    
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})
    
    user_data["mode"] = new_mode
    all_state[user_id] = user_data
    save_user_state(all_state)
    
    await message.answer(
        f"✅ Режим переключён на <b>{new_mode.upper()}</b>.\n\n"
        "‼️ <b>ТРЕБУЕТСЯ ПОЛНЫЙ ПЕРЕЗАПУСК БОТА</b>, чтобы изменения вступили в силу.",
        reply_markup=get_settings_menu_keyboard(),
        parse_mode=ParseMode.HTML
    )



@router.message(F.text == "Общий лимит")
async def trading_max_volume(message: Message, state: FSMContext):
    await message.answer("Введите новый общий торговый объём (например, 5000):")
    await state.set_state(MaxVolumeStates.waiting_max_volume)

@router.message(MaxVolumeStates.waiting_max_volume)
async def trading_set_max_volume(message: Message, state: FSMContext):
    try:
        max_vol = float(message.text.replace(",", "."))
        user_id = str(message.from_user.id)
        all_state = load_user_state()
        user_data = all_state.get(user_id, {})
        user_data["max_total_volume"] = max_vol
        all_state[user_id] = user_data
        save_user_state(all_state)
        await message.answer(f"✅ Общий лимит портфеля установлен: {max_vol}", reply_markup=get_settings_menu_keyboard())
        await state.clear()
    except ValueError:
        await message.answer("Неверный формат. Введите число.")

@router.message(F.text == "Лимит позиции")
async def trading_position_volume(message: Message, state: FSMContext):
    await message.answer("Введите новый объём для одной позиции (например, 1000):")
    await state.set_state(PositionVolumeStates.waiting_volume)

@router.message(PositionVolumeStates.waiting_volume)
async def trading_set_position_volume(message: Message, state: FSMContext):
    try:
        vol = float(message.text.replace(",", "."))
        user_id = str(message.from_user.id)
        all_state = load_user_state()
        user_data = all_state.get(user_id, {})
        user_data["volume"] = vol
        all_state[user_id] = user_data
        save_user_state(all_state)
        await message.answer(f"✅ Лимит позиции установлен: {vol}", reply_markup=get_settings_menu_keyboard())
        await state.clear()
    except ValueError:
        await message.answer("Неверный формат. Введите число.")

@router.message(F.text == "Стратегия торговли")
async def trading_mode_select(message: Message):
    rows = [[types.KeyboardButton(text=mode)] for mode in STRATEGY_MODES]
    rows.append([types.KeyboardButton(text="⬅️ Назад")])
    kb = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=rows)
    await message.answer("Выберите режим торговли:", reply_markup=kb)

@router.message(F.text.in_(set(STRATEGY_MODES)))
async def trading_mode_set(message: Message):
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})
    
    user_data["strategy_mode"] = message.text
    all_state[user_id] = user_data
    save_user_state(all_state)
    await message.answer(f"✅ Режим торговли переключён на {message.text}", reply_markup=get_settings_menu_keyboard())


@router.message(F.text == "Режим трейлинга")
async def trailing_mode_menu(message: Message):
    cfg = load_user_config(message.from_user.id)
    current = cfg.get("trailing_mode") or getattr(config, "ACTIVE_TRAILING_MODE", "adaptive")
    rows = [[types.KeyboardButton(text=mode)] for mode in TRAILING_MODES_AVAILABLE]
    rows.append([types.KeyboardButton(text="⬅️ Назад")])
    kb = ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True)
    await message.answer(
        f"Текущий режим: <b>{current}</b>\nВыберите режим трейлинга:",
        reply_markup=kb,
        parse_mode=ParseMode.HTML,
    )


@router.message(F.text.in_(set(TRAILING_MODES_AVAILABLE)))
async def trailing_mode_set(message: Message):
    new_mode = message.text.strip()
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})
    user_data["trailing_mode"] = new_mode
    all_state[user_id] = user_data
    save_user_state(all_state)
    await message.answer(
        f"✅ Режим трейлинга установлен: <b>{new_mode}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=get_settings_menu_keyboard(),
    )


@router.message(F.text.in_({"Трейлинг: Порог", "Трейлинг: Отступ"}))
async def trailing_metric_entry(message: Message, state: FSMContext):
    metric = "start" if "Порог" in message.text else "gap"
    await state.clear()
    await state.update_data(trailing_metric=metric)
    await message.answer(
        "Выберите стратегию, для которой хотите изменить параметр трейлинга:",
        reply_markup=get_trailing_logic_keyboard(message.from_user.id),
    )
    await state.set_state(TrailingConfigStates.waiting_logic)


@router.message(TrailingConfigStates.waiting_logic)
async def trailing_metric_choose_logic(message: Message, state: FSMContext):
    text = (message.text or "").strip()
    if text == "⬅️ Назад":
        await state.clear()
        await message.answer("Настройки трейлинга отменены.", reply_markup=get_settings_menu_keyboard())
        return

    cfg = load_user_config(message.from_user.id)
    if text == "Текущая стратегия":
        logic = cfg.get("strategy_mode", "full")
    else:
        logic = text

    data = await state.get_data()
    metric = data.get("trailing_metric", "start")
    await state.update_data(selected_logic=logic)

    prompt = (
        "Введите новый порог активации трейлинга в процентах (0.5 – 20):"
        if metric == "start"
        else "Введите новый отступ трейлинга в процентах (0.1 – 5):"
    )

    await message.answer(prompt, reply_markup=types.ReplyKeyboardRemove())
    await state.set_state(TrailingConfigStates.waiting_value)


@router.message(TrailingConfigStates.waiting_value)
async def trailing_metric_set_value(message: Message, state: FSMContext):
    text = (message.text or "").strip()
    if text.lower() in {"отмена", "cancel", "назад"}:
        await state.clear()
        await message.answer("Настройка трейлинга отменена.", reply_markup=get_settings_menu_keyboard())
        return

    try:
        value = float(text.replace(",", "."))
    except ValueError:
        await message.answer("Неверный формат. Введите число.")
        return

    data = await state.get_data()
    logic = data.get("selected_logic")
    metric = data.get("trailing_metric", "start")

    if not logic:
        await state.clear()
        await message.answer("Не удалось определить стратегию. Начните настройку заново.", reply_markup=get_settings_menu_keyboard())
        return

    if metric == "start" and not (0.5 <= value <= 20):
        await message.answer("Значение должно быть в диапазоне 0.5 – 20.")
        return
    if metric == "gap" and not (0.1 <= value <= 5):
        await message.answer("Значение должно быть в диапазоне 0.1 – 5.")
        return

    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})

    key = "trailing_start_pct" if metric == "start" else "trailing_gap_pct"
    mapping = user_data.get(key, {})
    mapping[logic] = value
    user_data[key] = mapping
    all_state[user_id] = user_data
    save_user_state(all_state)
    await state.clear()

    label = "Порог активации" if metric == "start" else "Отступ"
    await message.answer(
        f"✅ {label} для <b>{logic}</b> установлен на <b>{value:.2f}%</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=get_settings_menu_keyboard(),
    )

# --- Статус и управление ---
@router.message(F.text == "Открытые позиции")
async def bot_status(message: Message):
    summary = get_positions_summary(message.from_user.id)
    await message.answer(summary, reply_markup=get_status_menu_keyboard(), parse_mode="HTML")


@router.message(F.text == "Данные по кошельку")
async def info_wallet(message: Message):
    summary = get_wallet_summary(message.from_user.id)
    await message.answer(summary, reply_markup=get_status_menu_keyboard(), parse_mode="HTML")


@router.message(F.text == "Активные настройки")
async def info_settings_snapshot(message: Message):
    summary = format_settings_snapshot(message.from_user.id)
    await message.answer(summary, reply_markup=get_status_menu_keyboard(), parse_mode="HTML")


@router.message(F.text == "Режим тишины")
async def bot_sleep_toggle(message: Message):
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})
    current = user_data.get("sleep_mode", False)
    user_data["sleep_mode"] = not current
    all_state[user_id] = user_data
    save_user_state(all_state)
    status = "АКТИВИРОВАН" if user_data["sleep_mode"] else "ОТКЛЮЧЁН"
    await message.answer(f"Режим тишины {status}.", reply_markup=get_bot_control_keyboard())


@router.message(F.text == "ML фильтры")
async def ml_filters_toggle(message: Message):
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})
    strict_defaults = getattr(config, "ML_STRICT_FILTERS", {})
    strict_cfg = user_data.get("ML_STRICT_FILTERS") or {}
    current = bool(strict_cfg.get("ENABLED", strict_defaults.get("ENABLED", True)))
    new_state = not current
    strict_cfg["ENABLED"] = new_state
    user_data["ML_STRICT_FILTERS"] = strict_cfg
    all_state[user_id] = user_data
    save_user_state(all_state)
    min_ml = utils.safe_to_float(strict_cfg.get("MIN_WORKING_ML", strict_defaults.get("MIN_WORKING_ML", 0.33)))
    max_ml = utils.safe_to_float(strict_cfg.get("MAX_WORKING_ML", strict_defaults.get("MAX_WORKING_ML", 0.85)))
    status_text = "включены" if new_state else "выключены"
    await message.answer(
        f"ML фильтры {status_text.upper()}. Рабочий диапазон {min_ml:.2f}–{max_ml:.2f}.",
        reply_markup=get_bot_control_keyboard(),
    )


@router.message(F.text == "Остановить бот")
async def bot_stop(message: Message):
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})
    user_data["bot_active"] = False
    all_state[user_id] = user_data
    save_user_state(all_state)
    await message.answer("🔴 Бот остановлен.", reply_markup=get_bot_control_keyboard())


@router.message(F.text == "Запустить бот")
async def bot_start(message: Message):
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})
    user_data["bot_active"] = True
    all_state[user_id] = user_data
    save_user_state(all_state)
    await message.answer("✅ Торговая логика активирована.", reply_markup=get_bot_control_keyboard())


# --- Раздел «Отчёты» ---
@router.message(F.text == "PnL за 24ч")
async def info_trades_day(message: Message):
    await message.answer(
        format_pnl_change(24),
        reply_markup=get_reports_menu_keyboard(),
        parse_mode="HTML",
    )


@router.message(F.text == "PnL за 7д")
async def info_trades_week(message: Message):
    await message.answer(
        format_pnl_change(24 * 7),
        reply_markup=get_reports_menu_keyboard(),
        parse_mode="HTML",
    )


@router.message(F.text == "PnL за 30д")
async def info_trades_month(message: Message):
    await message.answer(
        format_pnl_change(24 * 30),
        reply_markup=get_reports_menu_keyboard(),
        parse_mode="HTML",
    )

# --- Раздел «Сервис» ---
@router.message(F.text == "Обратная связь")
async def service_feedback_start(message: Message, state: FSMContext):
    await message.answer("Опишите вашу проблему или предложение одним сообщением:")
    await state.set_state(FeedbackStates.waiting_message)

@router.message(FeedbackStates.waiting_message)
async def service_feedback_receive(message: Message, state: FSMContext):
    # Используем первый ID из админского списка для отправки
    if config.ADMIN_IDS:
        admin_chat_id = list(config.ADMIN_IDS)[0]
        try:
            await bot.send_message(admin_chat_id, f"Сообщение от {message.from_user.full_name} (ID: {message.from_user.id}):\n\n{message.text}")
            await message.answer("Спасибо за обратную связь!", reply_markup=get_service_menu_keyboard())
        except Exception as e:
            logger.error(f"Could not send feedback to admin {admin_chat_id}: {e}")
            await message.answer("Не удалось отправить сообщение администратору. Попробуйте позже.", reply_markup=get_service_menu_keyboard())
    else:
        await message.answer("Администратор не настроен. Ваше сообщение не доставлено.", reply_markup=get_service_menu_keyboard())
    await state.clear()

# --- Команды для настройки трейлинга ---
@router.message(Command("set_trailing"))
async def set_trailing_cmd(message: Message):
    args = (message.text or "").split()[1:]
    if not args:
        await message.answer(
            "Использование:\n"
            "<code>/set_trailing &lt;pct&gt;</code>\n"
            "<code>/set_trailing &lt;logic&gt; &lt;pct&gt;</code>\n"
            "<b>logic:</b> golden_only | liquidation_only | full",
            parse_mode="HTML"
        )
        return

    try:
        if len(args) == 1:
            pct = float(args[0])
            logic = None
        elif len(args) == 2:
            logic, pct_str = args
            if logic not in {"golden_only", "liquidation_only", "full"}:
                raise ValueError("Неизвестная логика.")
            pct = float(pct_str)
        else:
            raise ValueError("Слишком много аргументов.")

        if not (0.5 <= pct <= 20):
            raise ValueError("Значение должно быть в диапазоне 0.5–20 %.")

        user_id = str(message.from_user.id)
        all_state = load_user_state()
        ud = all_state.get(user_id, {})
        if logic is None:
            logic = ud.get("strategy_mode", "full")

        trailing_map = ud.get("trailing_start_pct", {})
        trailing_map[logic] = pct
        ud["trailing_start_pct"] = trailing_map
        all_state[user_id] = ud
        save_user_state(all_state)

        await message.answer(f"✅ Порог трейлинг-стопа для режима <b>{logic}</b> установлен на <b>{pct:.2f}%</b>", parse_mode="HTML")

    except ValueError as e:
        await message.answer(f"Ошибка: {e}")

@router.message(Command("set_gap"))
async def set_trailing_gap_cmd(message: Message):
    args = (message.text or "").split()[1:]
    if not args:
        await message.answer(
            "Использование:\n"
            "<code>/set_gap &lt;pct&gt;</code>\n"
            "<code>/set_gap &lt;logic&gt; &lt;pct&gt;</code>\n"
            "<b>logic:</b> golden_only | liquidation_only | full",
            parse_mode="HTML"
        )
        return

    try:
        if len(args) == 1:
            pct = float(args[0])
            logic = None
        elif len(args) == 2:
            logic, pct_str = args
            if logic not in {"golden_only", "liquidation_only", "full"}:
                raise ValueError("Неизвестная логика.")
            pct = float(pct_str)
        else:
            raise ValueError("Слишком много аргументов.")

        if not (0.1 <= pct <= 5):
            raise ValueError("Допустимый диапазон 0.1–5 %.")

        user_id = str(message.from_user.id)
        all_state = load_user_state()
        ud = all_state.get(user_id, {})
        if logic is None:
            logic = ud.get("strategy_mode", "full")

        gap_map = ud.get("trailing_gap_pct", {})
        gap_map[logic] = pct
        ud["trailing_gap_pct"] = gap_map
        all_state[user_id] = ud
        save_user_state(all_state)

        await message.answer(f"✅ Отступ трейлинг-стопа для <b>{logic}</b> теперь <b>{pct:.2f}%</b>", parse_mode="HTML")

    except ValueError as e:
        await message.answer(f"Ошибка: {e}")
