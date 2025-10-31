import json

import os
import pathlib

from aiogram import F, Router, types, Bot, Dispatcher
from aiogram.filters.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, ReplyKeyboardMarkup
from aiogram.filters import Command

import logging
from aiogram.enums import ParseMode

# from MultiuserBot_V16 import TradingBot         # aiogram-3

logger = logging.getLogger(__name__)



router = Router()
dp = Dispatcher()        # глобальный диспетчер для всего Telegram-бота
router_admin = Router()  # сюда добавите админ-хендлеры при необходимости

# Подключаем пользовательский и админский роутеры
dp.include_router(router)
dp.include_router(router_admin)



# ── bot instance fallback ────────────────────────────────────────────────────
# ── telegram Bot instance (loader.py optional) ─────────────────────────────
try:
    from loader import bot as loader_bot
    bot = loader_bot
except Exception:
    from aiogram import Bot
    BOT_TOKEN = '5833594543:AAE8BXe2G3tmvZTixAHGhGKzJIPN66P64a4'
    if not BOT_TOKEN:
        raise RuntimeError(
            "Set BOT_TOKEN env variable or create loader.py exporting 'bot'"
        )
    bot = Bot(BOT_TOKEN)

# ── simple JSON helpers (local user_state.json) ──────────────────────────────
STATE_PATH = pathlib.Path(__file__).with_name("user_state.json")

def load_user_state() -> dict:
    if STATE_PATH.exists():
        with STATE_PATH.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    return {}

def save_user_state(data: dict) -> None:
    with STATE_PATH.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)

# ── auxiliary data files ────────────────────────────────────────────────────
POSITIONS_PATH = pathlib.Path(__file__).with_name("open_positions.json")
WALLET_PATH    = pathlib.Path(__file__).with_name("wallet_state.json")
TRADES_PATH    = pathlib.Path(__file__).with_name("trades_history.json")

def _load_json(path: pathlib.Path):
    if path.exists():
        with path.open("r", encoding="utf-8") as fp:
            try:
                return json.load(fp)
            except json.JSONDecodeError:
                return None
    return None

def _user_slice(all_data: dict, user_id: int | str, default=None):
    return all_data.get(str(user_id), default)

def get_positions_summary() -> str:
    data = _load_json(POSITIONS_PATH) or []
    if not data:
        return "Нет открытых позиций."
    lines = ["🪙 Открытые позиции:"]
    total_pnl = 0.0
    for pos in data:
        sym   = pos.get("symbol")
        size  = pos.get("size")
        pnl   = pos.get("pnl", 0.0)
        total_pnl += pnl
        lines.append(f"• {sym}: {size}  |  PnL: {pnl:+.2f}$")
    lines.append(f"\nИтого PnL: {total_pnl:+.2f}$")
    return "\n".join(lines)

def get_wallet_summary(user_id: int) -> str:
    all_wallets = _load_json(WALLET_PATH) or {}
    wallet = _user_slice(all_wallets, user_id, {})
    if not wallet:
        return "Нет данных кошелька."
    bal   = wallet.get("totalEquity")       or wallet.get("total_equity", "—")
    avail = wallet.get("availableBalance")  or wallet.get("available_balance", "—")
    used  = wallet.get("usedMargin")        or wallet.get("used_margin", "—")
    return (
        "💰 <b>Состояние кошелька</b>\n"
        f"Equity: {bal}\n"
        f"Available: {avail}\n"
        f"Used margin: {used}"
    )

# ── /set_trailing <pct> [logic] ──────────────────────────────────────────────
@router.message(Command("set_trailing"))
async def set_trailing_cmd(message: Message):
    """
    /set_trailing 4.5
    /set_trailing golden_only 6
    """
    args = message.get_args().split()
    if not args:
        await message.answer(
            "Использование:\n"
            "/set_trailing <pct>\n"
            "/set_trailing <logic> <pct>\n"
            "где <logic> = golden_only | liquidation_only | full"
        )
        return

    # --- распарсим аргументы -------------------------------------------------
    if len(args) == 1:
        try:
            pct = float(args[0])
            logic = None                    # применить к активной стратегии
        except ValueError:
            await message.answer("Первый аргумент должен быть числом (процент).")
            return
    elif len(args) == 2:
        logic, pct_str = args
        if logic not in {"golden_only", "liquidation_only", "full"}:
            await message.answer(
                "Неизвестная логика. Допустимо: golden_only | liquidation_only | full"
            )
            return
        try:
            pct = float(pct_str)
        except ValueError:
            await message.answer("Второй аргумент должен быть числом (процент).")
            return
    else:
        await message.answer("Слишком много аргументов. См. /help")
        return

    if not (0.5 <= pct <= 20):
        await message.answer("Значение должно быть в диапазоне 0.5–20 %.")
        return

    # --- сохраним в user_state ----------------------------------------------
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    ud = all_state.get(user_id, {})

    # если логика не указана – берём текущий режим пользователя
    if logic is None:
        logic = ud.get("strategy_mode", "full")

    trailing_map = ud.get("trailing_start_pct", {})
    trailing_map[logic] = pct
    ud["trailing_start_pct"] = trailing_map
    all_state[user_id] = ud
    save_user_state(all_state)

    await message.answer(
        f"✅ Порог трейлинг-стопа для режима *{logic}* установлен на {pct:.2f} %",
        parse_mode="Markdown"
    )

# ── /set_gap <pct> [logic] ──────────────────────────────────────────────────
@router.message(Command("set_gap"))
async def set_trailing_gap_cmd(message: Message):
    """
    /set_gap 0.9
    /set_gap full 1.2
    """
    args = message.get_args().split()
    if not args:
        await message.answer(
            "Использование:\n"
            "/set_gap <pct>\n"
            "/set_gap <logic> <pct>\n"
            "где <logic> = golden_only | liquidation_only | full"
        )
        return

    # ---- разбор аргументов --------------------------------------------------
    if len(args) == 1:
        try:
            pct = float(args[0])
            logic = None
        except ValueError:
            await message.answer("Первый аргумент должен быть числом (процент).")
            return
    elif len(args) == 2:
        logic, pct_str = args
        if logic not in {"golden_only", "liquidation_only", "full"}:
            await message.answer("Неизвестная логика.")
            return
        try:
            pct = float(pct_str)
        except ValueError:
            await message.answer("Второй аргумент должен быть числом.")
            return
    else:
        await message.answer("Слишком много аргументов.")
        return

    if not (0.1 <= pct <= 5):
        await message.answer("Допустимый диапазон 0.1–5 %.")
        return

    # ---- сохраняем в user_state --------------------------------------------
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

    await message.answer(
        f"✅ Отступ трейлинг-стопа для *{logic}* теперь {pct:.2f} %",
        parse_mode="Markdown",
    )

def _trades_change(hours: int) -> str:
    trades = _load_json(TRADES_PATH) or []
    if not trades:
        return "Нет данных о сделках."
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    pnl = sum(t.get("pnl", 0.0) for t in trades
        if "timestamp" in t and datetime.fromisoformat(t["timestamp"]) >= cutoff)
    return f"Суммарный PnL за период: {pnl:+.2f}$"

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
        [types.KeyboardButton(text="⬅️ Назад")],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def get_bot_menu_keyboard():
    kb = [
        [types.KeyboardButton(text="Статус позиций")],
        [types.KeyboardButton(text="Режим тишины")],
        [types.KeyboardButton(text="Остановить бот")],
        [types.KeyboardButton(text="Запустить бот")],
        [types.KeyboardButton(text="Изменить трейлинг-стоп")],   # ← добавили
        [types.KeyboardButton(text="Изменить трейлинг-gap")],
        [types.KeyboardButton(text="⬅️ Назад")],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def get_info_menu_keyboard():
    kb = [
        [types.KeyboardButton(text="Данные по кошельку")],
        [types.KeyboardButton(text="Сделки за сутки")],
        [types.KeyboardButton(text="Сделки за неделю")],
        [types.KeyboardButton(text="Сделки за месяц")],
        [types.KeyboardButton(text="⬅️ Назад")],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def get_service_menu_keyboard():
    kb = [
        [types.KeyboardButton(text="Обратная связь")],
        [types.KeyboardButton(text="⬅️ Назад")],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)


class VolumeStates(StatesGroup):
    waiting_volume = State()


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


@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("Главное меню", reply_markup=get_main_menu_keyboard())


# ── root menu ────────────────────────────────────────────────────────────────
@router.message(F.text == "Торговля")
async def show_trading_menu(message: Message):
    await message.answer("Раздел «Торговля»", reply_markup=get_trading_menu_keyboard())

@router.message(F.text == "Бот")
async def show_bot_menu(message: Message):
    await message.answer(
        "Раздел «Бот»",
        reply_markup=get_bot_menu_keyboard()
    )

@router.message(F.text == "Информация")
async def show_info_menu(message: Message):
    await message.answer("Раздел «Информация»", reply_markup=get_info_menu_keyboard())

@router.message(F.text == "Сервис")
async def show_service_menu(message: Message):
    await message.answer("Раздел «Сервис»", reply_markup=get_service_menu_keyboard())

@router.message(F.text == "⬅️ Назад")
async def back_to_main(message: Message):
    await message.answer("Главное меню", reply_markup=get_main_menu_keyboard())


# ── «Торговля» ───────────────────────────────────────────────────────────────
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

@router.message(F.text == "Общий торговый объём")
async def trading_max_volume(message: Message, state: FSMContext):
    await message.answer("Введите новый общий торговый объём:")
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
        await message.answer(f"✅ max_total_volume установлено: {max_vol}", reply_markup=get_trading_menu_keyboard())
        await state.clear()
    except ValueError:
        await message.answer("Неверный формат. Введите число.")

@router.message(F.text == "Объём торговой позиции")
async def trading_position_volume(message: Message, state: FSMContext):
    await message.answer("Введите новый объём позиции:")
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
        await message.answer(f"✅ volume установлен: {vol}", reply_markup=get_trading_menu_keyboard())
        await state.clear()
    except ValueError:
        await message.answer("Неверный формат. Введите число.")

@router.message(F.text == "Режим торговли")
async def trading_mode_select(message: Message):
    kb = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[
        [types.KeyboardButton(text="golden_only")],
        [types.KeyboardButton(text="liquidation_only")],
        [types.KeyboardButton(text="full")],
        [types.KeyboardButton(text="⬅️ Назад")],
    ])
    await message.answer("Выберите режим торговли:", reply_markup=kb)

@router.message(lambda m: m.text in {"golden_only", "liquidation_only", "full"})
async def trading_mode_set(message: Message):
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})
    user_data["strategy_mode"] = message.text
    all_state[user_id] = user_data
    save_user_state(all_state)
    await message.answer(f"✅ strategy_mode переключён на {message.text}", reply_markup=get_trading_menu_keyboard())


# ── «Бот» ─────────────────────────────────────────────────────────────────────
@router.message(F.text == "Статус позиций")
async def bot_status(message: Message):
    summary = get_positions_summary()
    await message.answer(summary, reply_markup=get_bot_menu_keyboard(), parse_mode="Markdown")

@router.message(F.text == "Режим тишины")
async def bot_sleep_toggle(message: Message):
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})
    current = user_data.get("sleep_mode", False)
    user_data["sleep_mode"] = not current
    all_state[user_id] = user_data
    save_user_state(all_state)
    status = "активирован" if user_data["sleep_mode"] else "отключён"
    await message.answer(f"Режим тишины {status}.", reply_markup=get_bot_menu_keyboard())

@router.message(F.text == "Остановить бот")
async def bot_stop(message: Message):
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})
    user_data["bot_active"] = False
    all_state[user_id] = user_data
    save_user_state(all_state)
    await message.answer("Бот остановлен.", reply_markup=get_bot_menu_keyboard())

@router.message(F.text == "Запустить бот")
async def bot_start(message: Message):
    user_id = str(message.from_user.id)
    all_state = load_user_state()
    user_data = all_state.get(user_id, {})
    user_data["bot_active"] = True
    all_state[user_id] = user_data
    save_user_state(all_state)
    await message.answer("Торговая логика активирована.", reply_markup=get_bot_menu_keyboard())

# # ───────── Toggle averaging via Telegram ─────────
# @router.message(Command("avg_on"))
# async def bot_avg_on(msg: types.Message):
#     uid = msg.from_user.id
#     # 1) включаем у живого TradingBot (если есть)
#     for bot in TradingBot:
#         if bot.user_id == uid:
#             bot.averaging_enabled = True
#             break

#     # 2) сохраняем в user_state.json
#     all_state = load_user_state()
#     ud = all_state.get(str(uid), {})
#     ud["averaging_enabled"] = True
#     all_state[str(uid)] = ud
#     save_user_state(all_state)

#     await msg.answer("✅ Усреднение ВКЛЮЧЕНО.", parse_mode=ParseMode.HTML)
#     logger.info("[avg_on] user %s enabled averaging", uid)

# @router.message(Command("avg_off"))
# async def bot_avg_off(msg: types.Message):
#     uid = msg.from_user.id
#     for bot in TradingBot:
#         if bot.user_id == uid:
#             bot.averaging_enabled = False
#             break

#     all_state = load_user_state()
#     ud = all_state.get(str(uid), {})
#     ud["averaging_enabled"] = False
#     all_state[str(uid)] = ud
#     save_user_state(all_state)

#     await msg.answer("⛔️ Усреднение ВЫКЛЮЧЕНО.", parse_mode=ParseMode.HTML)
#     logger.info("[avg_off] user %s disabled averaging", uid)


# ── «Информация» ──────────────────────────────────────────────────────────────
@router.message(F.text(equals="данные по кошельку", ignore_case=True))
async def info_wallet(message: Message):
    summary = get_wallet_summary(message.from_user.id)
    await message.answer(summary, reply_markup=get_info_menu_keyboard(), parse_mode="HTML")

@router.message(F.text == "Сделки за сутки")
async def info_trades_day(message: Message):
    await message.answer(_trades_change(24), reply_markup=get_info_menu_keyboard())

@router.message(F.text == "Сделки за неделю")
async def info_trades_week(message: Message):
    await message.answer(_trades_change(24*7), reply_markup=get_info_menu_keyboard())

@router.message(F.text == "Сделки за месяц")
async def info_trades_month(message: Message):
    await message.answer(_trades_change(24*30), reply_markup=get_info_menu_keyboard())


# ── «Сервис» ──────────────────────────────────────────────────────────────────
@router.message(F.text == "Обратная связь")
async def service_feedback_start(message: Message, state: FSMContext):
    await message.answer("Опишите вашу проблему или предложение одним сообщением:")
    await state.set_state(FeedbackStates.waiting_message)

@router.message(FeedbackStates.waiting_message)
async def service_feedback_receive(message: Message, state: FSMContext):
    admin_chat_id = "admin_id"  # замените на настоящий ID
    await bot.send_message(admin_chat_id, f"Сообщение от {message.from_user.id}:\n{message.text}")
    await message.answer("Спасибо за обратную связь!", reply_markup=get_service_menu_keyboard())
    await state.clear()