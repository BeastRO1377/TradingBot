
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.utils.keyboard import ReplyKeyboardMarkup
import json

bot = Bot(token="YOUR_TELEGRAM_BOT_TOKEN", parse_mode="HTML")
dp = Dispatcher(storage=MemoryStorage())

USER_STATE_FILE = "user_state.json"

def load_user_state():
    with open(USER_STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_user_state(data):
    with open(USER_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

class RegisterStates(StatesGroup):
    waiting_api_key = State()
    waiting_api_secret = State()

class VolumeStates(StatesGroup):
    waiting_volume = State()

@dp.message(commands=["start"])
async def cmd_start(message: Message):
    await message.answer("Добро пожаловать! Используй /register для начала.")

@dp.message(commands=["register"])
async def cmd_register(message: Message, state: FSMContext):
    await message.answer("Введите ваш API_KEY:")
    await state.set_state(RegisterStates.waiting_api_key)

@dp.message(RegisterStates.waiting_api_key)
async def reg_api_key(message: Message, state: FSMContext):
    await state.update_data(api_key=message.text.strip())
    await message.answer("Теперь введите API_SECRET:")
    await state.set_state(RegisterStates.waiting_api_secret)

@dp.message(RegisterStates.waiting_api_secret)
async def reg_api_secret(message: Message, state: FSMContext):
    data = await state.get_data()
    user_id = str(message.from_user.id)
    user_state = load_user_state()
    user_state[user_id] = {
        "banned": False,
        "volume": 0.01,
        "strategy": "macd",
        "registered": True,
        "api_key": data["api_key"],
        "api_secret": message.text.strip()
    }
    save_user_state(user_state)
    await message.answer("Вы успешно зарегистрированы!")
    await state.clear()

@dp.message(commands=["set_volume"])
async def cmd_set_volume(message: Message, state: FSMContext):
    await message.answer("Введите желаемый объём позиции:")
    await state.set_state(VolumeStates.waiting_volume)

@dp.message(VolumeStates.waiting_volume)
async def set_volume(message: Message, state: FSMContext):
    try:
        vol = float(message.text.strip())
        user_id = str(message.from_user.id)
        state_data = load_user_state()
        if user_id in state_data:
            state_data[user_id]["volume"] = vol
            save_user_state(state_data)
            await message.answer(f"Объём позиции установлен: {vol}")
        await state.clear()
    except Exception:
        await message.answer("Неверный формат. Повторите ввод.")

@dp.message(commands=["strategy"])
async def cmd_strategy(message: Message):
    kb = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[
        [types.KeyboardButton(text="macd")],
        [types.KeyboardButton(text="rsi")],
        [types.KeyboardButton(text="supertrend")],
        [types.KeyboardButton(text="drift")],
        [types.KeyboardButton(text="ml")],
        [types.KeyboardButton(text="golden")]
    ])
    await message.answer("Выберите стратегию:", reply_markup=kb)

@dp.message()
async def set_strategy(message: Message):
    if message.text in {"macd", "rsi", "supertrend", "drift", "ml", "golden"}:
        user_id = str(message.from_user.id)
        state_data = load_user_state()
        if user_id in state_data:
            state_data[user_id]["strategy"] = message.text
            save_user_state(state_data)
            await message.answer(f"Стратегия переключена на: {message.text}")
            if not state_data[user_id].get("quiet_mode", False):
                try:
                    await bot.send_message(chat_id=user_id, text=f"⚙️ Ваша стратегия теперь: {message.text}")
                except Exception:
                    pass
    elif message.text.startswith("/status"):
        user_id = str(message.from_user.id)
        state_data = load_user_state()
        if user_id in state_data:
            user = state_data[user_id]
            await message.answer(f"Стратегия: {user['strategy']}, объём: {user['volume']}, бан: {user['banned']}")

@dp.message(commands=["admin_stop"])
async def admin_stop(message: Message):
    if str(message.from_user.id) != "admin_id":
        await message.answer("Недостаточно прав.")
        return
    await message.answer("🛑 Бот остановлен администратором. Все пользователи уведомлены.")

@dp.message(commands=["admin_ban", "admin_unban"])
async def admin_ban_unban(message: Message):
    if str(message.from_user.id) != "admin_id":
        return
    cmd, uid = message.text.split()
    state_data = load_user_state()
    if uid in state_data:
        state_data[uid]["banned"] = (cmd == "/admin_ban")
        save_user_state(state_data)
        await message.answer(f"Пользователь {uid} {'забанен' if cmd == '/admin_ban' else 'разбанен'}.")

@dp.message(commands=["admin_set"])
async def admin_set(message: Message):
    if str(message.from_user.id) != "admin_id":
        return
    try:
        _, uid, param, val = message.text.split()
        state_data = load_user_state()
        if uid in state_data:
            if param in state_data[uid]:
                state_data[uid][param] = float(val) if param == "volume" else val
                save_user_state(state_data)
                await message.answer(f"{param} пользователя {uid} установлен в {val}")
    except Exception as e:
        await message.answer(f"Ошибка: {e}")

@dp.message(lambda msg: msg.text.startswith("/statement_"))
async def admin_statement(message: Message):
    if str(message.from_user.id) != "admin_id":
        return
    uid = message.text.replace("/statement_", "")
    state_data = load_user_state()
    if uid in state_data:
        user = state_data[uid]
        await message.answer(f"📊 Выписка пользователя {uid}:
Стратегия: {user['strategy']}, объём: {user['volume']}, бан: {user['banned']}")
