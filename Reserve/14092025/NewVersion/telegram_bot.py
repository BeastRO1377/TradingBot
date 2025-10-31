# telegram_bot.py
import json
import pathlib
import logging
import time

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

def get_positions_summary(user_id: int) -> str:
    all_positions = _load_json(config.OPEN_POS_JSON)
    user_positions = _user_slice(all_positions, user_id, [])
    
    if not user_positions:
        return "Нет открытых позиций."
        
    lines = ["🪙 <b>Открытые позиции:</b>"]
    total_pnl = 0.0
    for pos in user_positions:
        sym = pos.get("symbol")
        size = pos.get("size")
        pnl = utils.safe_to_float(pos.get("pnl", 0.0))
        total_pnl += pnl
        lines.append(f"• <code>{sym:<10}</code> | Vol: {size} | PnL: {pnl:+.2f}$")
    lines.append(f"\n<b>Итого PnL: {total_pnl:+.2f}$</b>")
    return "\n".join(lines)

def get_wallet_summary(user_id: int) -> str:
    all_wallets = _load_json(config.WALLET_JSON)
    wallet = _user_slice(all_wallets, user_id, {})
    if not wallet:
        return "Нет данных кошелька."
    bal = wallet.get("totalEquity", "—")
    avail = wallet.get("availableBalance", "—")
    used = wallet.get("usedMargin", "—")
    return (
        "💰 <b>Состояние кошелька</b>\n"
        f"Equity: <code>{bal}</code>\n"
        f"Available: <code>{avail}</code>\n"
        f"Used margin: <code>{used}</code>"
    )

def _trades_change(hours: int, user_id: int) -> str:
    all_trades = _load_json(config.TRADES_JSON)
    user_trades = _user_slice(all_trades, user_id, [])
    if not user_trades:
        return "Нет данных о сделках."
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    pnl = sum(
        utils.safe_to_float(t.get("pnl", 0.0)) for t in user_trades
        if "timestamp" in t and datetime.fromisoformat(t["timestamp"].split('.')[0]) >= cutoff
    )
    return f"Суммарный PnL за {hours}ч: <b>{pnl:+.2f}$</b>"

# --- Клавиатуры ---

def get_main_menu_keyboard():
    kb = [
        [types.KeyboardButton(text="Торговля"), types.KeyboardButton(text="Бот")],
        [types.KeyboardButton(text="Информация"), types.KeyboardButton(text="Сервис")],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def get_trading_menu_keyboard():
    kb = [
        [types.KeyboardButton(text="Обновление API")],
        [types.KeyboardButton(text="Общий торговый объём")],
        [types.KeyboardButton(text="Объём торговой позиции")],
        [types.KeyboardButton(text="Режим торговли")],
        [types.KeyboardButton(text="Режим")],
        [types.KeyboardButton(text="⬅️ Назад")],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def get_bot_menu_keyboard():
    kb = [
        [types.KeyboardButton(text="Статус позиций")],
        [types.KeyboardButton(text="Режим тишины")],
        [types.KeyboardButton(text="Остановить бот"), types.KeyboardButton(text="Запустить бот")],
        [types.KeyboardButton(text="Изменить трейлинг-стоп"), types.KeyboardButton(text="Изменить трейлинг-gap")],
        [types.KeyboardButton(text="⬅️ Назад")],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def get_info_menu_keyboard():
    kb = [
        [types.KeyboardButton(text="Данные по кошельку")],
        [types.KeyboardButton(text="Сделки за сутки")],
        [types.KeyboardButton(text="Сделки за неделю"), types.KeyboardButton(text="Сделки за месяц")],
        [types.KeyboardButton(text="⬅️ Назад")],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def get_service_menu_keyboard():
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

class StrategyModeStates(StatesGroup):
    waiting_mode = State()

class FeedbackStates(StatesGroup):
    waiting_message = State()

# --- Обработчики команд и меню ---

@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("Главное меню", reply_markup=get_main_menu_keyboard())

# --- Навигация по главному меню ---
@router.message(F.text == "Торговля")
async def show_trading_menu(message: Message):
    await message.answer("Раздел «Торговля»", reply_markup=get_trading_menu_keyboard())

@router.message(F.text == "Бот")
async def show_bot_menu(message: Message):
    await message.answer("Раздел «Бот»", reply_markup=get_bot_menu_keyboard())

@router.message(F.text == "Информация")
async def show_info_menu(message: Message):
    await message.answer("Раздел «Информация»", reply_markup=get_info_menu_keyboard())

@router.message(F.text == "Сервис")
async def show_service_menu(message: Message):
    await message.answer("Раздел «Сервис»", reply_markup=get_service_menu_keyboard())

@router.message(F.text == "⬅️ Назад")
async def back_to_main(message: Message):
    await message.answer("Главное меню", reply_markup=get_main_menu_keyboard())

# --- Раздел «Торговля» ---
@router.message(F.text == "Обновление API")
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
    await message.answer("✅ Ключи API обновлены.", reply_markup=get_trading_menu_keyboard())
    await state.clear()

@router.message(F.text == "Режим")
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
        reply_markup=get_trading_menu_keyboard(),
        parse_mode=ParseMode.HTML
    )



@router.message(F.text == "Общий торговый объём")
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
        await message.answer(f"✅ Общий торговый объём установлен: {max_vol}", reply_markup=get_trading_menu_keyboard())
        await state.clear()
    except ValueError:
        await message.answer("Неверный формат. Введите число.")

@router.message(F.text == "Объём торговой позиции")
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
        await message.answer(f"✅ Объём позиции установлен: {vol}", reply_markup=get_trading_menu_keyboard())
        await state.clear()
    except ValueError:
        await message.answer("Неверный формат. Введите число.")

@router.message(F.text == "Режим торговли")
async def trading_mode_select(message: Message):
    kb = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[
        [types.KeyboardButton(text="full")],
        [types.KeyboardButton(text="golden_squeeze")],
        [types.KeyboardButton(text="squeeze_only")], # <-- НОВАЯ КНОПКА
        [types.KeyboardButton(text="golden_only")],
        [types.KeyboardButton(text="liquidation_only")],
        [types.KeyboardButton(text="⬅️ Назад")],

    ])
    await message.answer("Выберите режим торговли:", reply_markup=kb)

@router.message(F.text.in_({"golden_only", "liquidation_only", "full", "squeeze_only", "golden_squeeze"}))
async def trading_mode_set(message: Message):
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})
    
    # --- ИЗМЕНЕНИЕ ---
    # Мы добавили "squeeze_only" в список режимов в декораторе выше,
    # поэтому этот код будет работать для него автоматически.
    # Добавляем "golden_squeeze" на всякий случай, если его там не было.
    user_data["strategy_mode"] = message.text
    all_state[user_id] = user_data
    save_user_state(all_state)
    await message.answer(f"✅ Режим торговли переключён на {message.text}", reply_markup=get_trading_menu_keyboard())

# --- Раздел «Бот» ---
@router.message(F.text == "Статус позиций")
async def bot_status(message: Message):
    summary = get_positions_summary(message.from_user.id)
    await message.answer(summary, reply_markup=get_bot_menu_keyboard(), parse_mode="HTML")

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
    await message.answer(f"Режим тишины {status}.", reply_markup=get_bot_menu_keyboard())

@router.message(F.text == "Остановить бот")
async def bot_stop(message: Message):
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})
    user_data["bot_active"] = False
    all_state[user_id] = user_data
    save_user_state(all_state)
    await message.answer("🔴 Бот остановлен.", reply_markup=get_bot_menu_keyboard())

@router.message(F.text == "Запустить бот")
async def bot_start(message: Message):
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})
    user_data["bot_active"] = True
    all_state[user_id] = user_data
    save_user_state(all_state)
    await message.answer("✅ Торговая логика активирована.", reply_markup=get_bot_menu_keyboard())

# --- Раздел «Информация» ---
@router.message(F.text.lower() == "данные по кошельку")
async def info_wallet(message: Message):
    summary = get_wallet_summary(message.from_user.id)
    await message.answer(summary, reply_markup=get_info_menu_keyboard(), parse_mode="HTML")

@router.message(F.text == "Сделки за сутки")
async def info_trades_day(message: Message):
    await message.answer(_trades_change(24, message.from_user.id), reply_markup=get_info_menu_keyboard(), parse_mode="HTML")

@router.message(F.text == "Сделки за неделю")
async def info_trades_week(message: Message):
    await message.answer(_trades_change(24*7, message.from_user.id), reply_markup=get_info_menu_keyboard(), parse_mode="HTML")

@router.message(F.text == "Сделки за месяц")
async def info_trades_month(message: Message):
    await message.answer(_trades_change(24*30, message.from_user.id), reply_markup=get_info_menu_keyboard(), parse_mode="HTML")

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