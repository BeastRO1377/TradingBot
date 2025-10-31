# ai_ml.py
import logging
import json
import re
import asyncio
from typing import Dict, Any

import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import mlx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim
from safetensors.numpy import save_file as save_safetensors, load_file as load_safetensors
from openai import AsyncOpenAI

import bot_core
import config
from utils import safe_to_float, safe_parse_json, compute_pct

logger = logging.getLogger(__name__)

# Константы для анализа ответов AI
NEGATIVE_CUES = {
    "негатив", "опас", "перегрет", "переоцен", "risk", "bear", "downtrend",
    "слабый спрос", "дистрибуц", "падени", "разворот вниз", "sell-off", "dump"
}

# --- Классы моделей и инференса ---

class GoldenNetMLX(mlx_nn.Module):
    """Нейронная сеть на MLX для предсказания качества входа."""
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = mlx_nn.Linear(input_size, hidden_size)
        self.bn1 = mlx_nn.BatchNorm(hidden_size)
        self.fc2 = mlx_nn.Linear(hidden_size, hidden_size)
        self.bn2 = mlx_nn.BatchNorm(hidden_size)
        self.fc3 = mlx_nn.Linear(hidden_size, 1)
        self.dropout = mlx_nn.Dropout(0.2)

    def __call__(self, x):
        x = mlx_nn.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = mlx_nn.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

    def state_dict_numpy(self) -> dict:
        """Собирает веса слоёв в dict numpy-массивов для safetensors."""
        to_np = lambda t: np.array(t)
        sd = {
            "fc1.weight": to_np(self.fc1.weight), "fc1.bias": to_np(self.fc1.bias),
            "bn1.weight": to_np(self.bn1.weight), "bn1.bias": to_np(self.bn1.bias),
            "bn1.running_mean": to_np(self.bn1.running_mean), "bn1.running_var": to_np(self.bn1.running_var),
            "fc2.weight": to_np(self.fc2.weight), "fc2.bias": to_np(self.fc2.bias),
            "bn2.weight": to_np(self.bn2.weight), "bn2.bias": to_np(self.bn2.bias),
            "bn2.running_mean": to_np(self.bn2.running_mean), "bn2.running_var": to_np(self.bn2.running_var),
            "fc3.weight": to_np(self.fc3.weight), "fc3.bias": to_np(self.fc3.bias),
        }
        return sd

    def load_weights(self, path: str):
        """Загружает веса из safetensors в слои MLX."""
        tensors = load_safetensors(path)
        to_mlx = mlx.core.array
        self.fc1.weight, self.fc1.bias = to_mlx(tensors["fc1.weight"]), to_mlx(tensors["fc1.bias"])
        self.bn1.weight, self.bn1.bias = to_mlx(tensors["bn1.weight"]), to_mlx(tensors["bn1.bias"])
        self.bn1.running_mean, self.bn1.running_var = to_mlx(tensors["bn1.running_mean"]), to_mlx(tensors["bn1.running_var"])
        self.fc2.weight, self.fc2.bias = to_mlx(tensors["fc2.weight"]), to_mlx(tensors["fc2.bias"])
        self.bn2.weight, self.bn2.bias = to_mlx(tensors["bn2.weight"]), to_mlx(tensors["bn2.bias"])
        self.bn2.running_mean, self.bn2.running_var = to_mlx(tensors["bn2.running_mean"]), to_mlx(tensors["bn2.running_var"])
        self.fc3.weight, self.fc3.bias = to_mlx(tensors["fc3.weight"]), to_mlx(tensors["fc3.bias"])

class MLXInferencer:
    """Класс для загрузки модели, скейлера и выполнения предсказаний."""
    def __init__(self, model_path=config.ML_MODEL_PATH, scaler_path=config.SCALER_PATH):
        self.model = None
        self.scaler = None
        if Path(model_path).exists():
            try:
                self.model = GoldenNetMLX(input_size=len(config.FEATURE_KEYS))
                self.model.load_weights(str(model_path))
                self.model.eval()
                logger.info(f"Модель из {model_path} успешно загружена.")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели {model_path}: {e}", exc_info=True)
        if Path(scaler_path).exists():
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Скейлер из {scaler_path} успешно загружен.")
            except Exception as e:
                logger.error(f"Ошибка загрузки скейлера {scaler_path}: {e}", exc_info=True)

    def infer(self, features: np.ndarray) -> np.ndarray:
        if self.model is None: return np.array([[0.0]])
        if self.scaler:
            features = self.scaler.transform(features)
        prediction = self.model(mlx.core.array(features))
        return np.array(prediction)

# --- Функции обучения и сохранения ---

def train_golden_model_mlx(training_data, num_epochs: int = 30, lr: float = 1e-3):
    logger.info("[MLX] Запуск обучения на MLX...")
    feats = np.asarray([d["features"] for d in training_data], dtype=np.float32)
    targ = np.asarray([d["target"] for d in training_data], dtype=np.float32)
    mask = ~(np.isnan(feats).any(1) | np.isinf(feats).any(1))
    feats, targ = feats[mask], targ[mask]
    if feats.size == 0:
        raise ValueError("train_golden_model_mlx: нет валидных сэмплов")

    scaler = StandardScaler().fit(feats)
    feats_scaled = scaler.transform(feats).astype(np.float32)
    targ = targ.reshape(-1, 1)

    model = GoldenNetMLX(input_size=feats_scaled.shape[1])
    optimizer = mlx_optim.Adam(learning_rate=lr)
    loss_fn = lambda m, x, y: mlx_nn.losses.mse_loss(m(x), y).mean()
    loss_and_grad_fn = mlx_nn.value_and_grad(model, loss_fn)

    for epoch in range(num_epochs):
        x_train, y_train = mlx.core.array(feats_scaled), mlx.core.array(targ)
        loss, grads = loss_and_grad_fn(model, x_train, y_train)
        optimizer.update(model, grads)
        mlx.core.eval(model.parameters(), optimizer.state)
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1} [MLX] – Loss: {loss.item():.5f}")

    return model, scaler

def save_mlx_checkpoint(model: GoldenNetMLX, scaler: StandardScaler,
                        model_path: str = str(config.ML_MODEL_PATH),
                        scaler_path: str = str(config.SCALER_PATH)):
    tensors = model.state_dict_numpy()
    save_safetensors(tensors, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Модель сохранена → {model_path}; scaler → {scaler_path}")

# --- Взаимодействие с AI ---

def _ask_ollama_json_sync(model: str, messages: list[dict], base_url: str, timeout_s: float) -> Dict[str, Any]:
    """
    СИНХРОННАЯ версия функции для выполнения в отдельном потоке.
    """
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="ollama", timeout=timeout_s - 1)
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, response_format={"type": "json_object"}, temperature=0.2
        )
        raw = (resp.choices[0].message.content or "").strip()
        return safe_parse_json(raw, default={"action": "REJECT", "justification": "bad json"})
    except Exception as e:
        logger.error(f"Ошибка в синхронном запросе к модели {model}: {e}")
        if "timed out" in str(e).lower():
             return {"action": "REJECT", "justification": f"AI HTTP Timeout Error: {e}"}
        return {"action": "REJECT", "justification": f"AI Sync Error: {e}"}


async def ask_ollama_json(model: str, messages: list[dict], timeout_s: float, base_url: str) -> Dict[str, Any]:
    """
    АСИНХРОННАЯ обертка, которая выполняет тяжелый, блокирующий запрос в отдельном потоке.
    """
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_ask_ollama_json_sync, model, messages, base_url, timeout_s),
            timeout=timeout_s
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"Общий таймаут ({timeout_s}с) при запросе к модели {model}.")
        return {"action": "REJECT", "justification": f"Global Timeout ({timeout_s}s)"}
    except Exception as e:
        logger.error(f"Неожиданная ошибка в async wrapper для {model}: {e}", exc_info=True)
        return {"action": "REJECT", "justification": f"AI Wrapper Error: {e}"}



# --- Формирование промптов ---

def _format(v, spec, na="N/A"):
    try:
        x = float(v)
        if np.isinf(x) or np.isnan(x): return na
        return f"{x:{spec}}"
    except (ValueError, TypeError):
        return na


def build_primary_prompt(candidate: dict, features: dict, shared_ws) -> str:
    """
    [ИСПРАВЛЕННАЯ ВЕРСИЯ V12] Единый, унифицированный промпт,
    корректно работающий со всеми типами сигналов.
    """
    sym = candidate.get("symbol", "UNKNOWN")
    side = str(candidate.get("side", "Buy")).upper()
    source = candidate.get("source", "unknown")
    source_title = source.replace("_", " ").title()

    # --- Подготовка ключевых данных для анализа ---
    rsi_val = safe_to_float(features.get('rsi14'))
    adx_val = safe_to_float(features.get('adx14'))
    pct_5m = safe_to_float(features.get('pct5m'))
    pct_30m = safe_to_float(features.get('pct30m'))
    vol_anomaly = features.get('volume_anomaly', 1.0)
    
    prompt_header = "SYSTEM: Ты — AI риск-менеджер Plutus. Твоя задача — дать четкий и однозначный вердикт по торговому сигналу: EXECUTE или REJECT. Твой ответ — всегда только валидный JSON."
    
    # --- НАЧАЛО ИСПРАВЛЕНИЯ: Этот блок теперь является общим для всех ---
    prompt_data = f"""
    USER:
    **1. СИГНАЛ НА УТВЕРЖДЕНИЕ:**
    - Инструмент: {sym}, Направление: {side}, Стратегия: {source_title}

    **2. КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ:**
    - Импульс (5м): {_format(pct_5m, '.2f')}%
    - Контекст (30м): {_format(pct_30m, '.2f')}%
    - RSI(14): {_format(rsi_val, '.1f')}
    - ADX(14) (Сила тренда): {_format(adx_val, '.1f')}
    - Аномалия объема (1m): {_format(vol_anomaly, '.1f')}x
    """
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    prompt_task = ""
    if 'squeeze' in source:
        # --- НАЧАЛО ИЗМЕНЕНИЯ ---
        prompt_task = """
        **3. ПРАВИЛА АНАЛИЗА (Squeeze - Контртренд):**
        Наша цель — войти ПРОТИВ импульса в момент его ИСТОЩЕНИЯ.
        
        **Шаг 1: Проверка RSI.**
        - Для 'SELL', RSI должен быть экстремально высоким (> 75).
        - Для 'BUY', RSI должен быть экстремально низким (< 25).
        
        **Шаг 2: КРИТИЧЕСКАЯ ПРОВЕРКА СИЛЫ ТРЕНДА по ADX.**
        - Если ADX > 45: Это ОЧЕНЬ СИЛЬНЫЙ ТРЕНД, а не истощение. Вероятность продолжения движения высока. **Верни "REJECT".** Обоснование: "ADX слишком высокий для контртренда".
        - Если ADX < 20: Тренд слабый, разворот маловероятен. **Верни "REJECT".** Обоснование: "ADX слишком низкий, нет импульса для разворота".
        
        **Шаг 3: Принятие решения.**
        - **Верни "EXECUTE" только если выполнены ОБА условия:**
          1. RSI находится в экстремальной зоне (Шаг 1).
          2. ADX находится в "зоне отката" (между 20 и 45).
        - Во всех остальных случаях верни "REJECT".
        """
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    elif 'liquidation' in source:
        # Здесь мы используем скоринг, как и договаривались
        prompt_task = f"""
        **3. АЛГОРИТМ СКОРИНГА (Liquidation Cascade - Контртренд):**
        **Цель:** Войти против каскада ликвидаций.
        **Шаг 1: Базовые баллы (Порог: 75).**
        - Начисли **50 баллов** по умолчанию.
        **Шаг 2: Бонусные баллы (суммируются).**
        - Если `Аномалия объема` > 4.0x: **+20 баллов**.
        - Если RSI экстремальный (для 'BUY' < 25; для 'SELL' > 75): **+15 баллов**.
        - Если ADX > 45: **+10 баллов**.
        **Шаг 3: Финальный вердикт.**
        - **ЕСЛИ итоговый балл >= 75:** Вернуть "EXECUTE".
        - **ИНАЧЕ:** Вернуть "REJECT".
        """
    else:  # Golden Setup по умолчанию
        # --- НОВАЯ, УМНАЯ ЛОГИКА ДЛЯ GOLDEN SETUP ---
        prompt_task = f"""
        **3. АЛГОРИТМ АНАЛИЗА (Golden Setup - "Сжатая пружина"):**
        **Контекст:** Обнаружен паттерн накопления (рост ОИ, всплеск объема). Это сигнал ПРОДОЛЖЕНИЯ тренда.
        
        **КРИТИЧЕСКИЙ ШАГ: Проверка контекста предшествующего движения.**
        Твоя главная задача — не попасть в ловушку на пике или дне рынка. Используй показатель `Контекст (30м)`.

        - **ЕСЛИ сигнал 'Buy', а `Контекст (30м)` уже показывает сильный рост (например, > 3.5%):**
          Это очень плохой знак. Вероятно, это не накопление, а распределение на пике. **Верни "REJECT".** Обоснование: "Высокий риск покупки на локальном максимуме".

        - **ЕСЛИ сигнал 'Sell', а `Контекст (30м)` уже показывает сильное падение (например, < -3.5%):**
          Это очень плохой знак. Вероятно, это не набор шорта, а паническая продажа на дне. **Верни "REJECT".** Обоснование: "Высокий риск продажи на локальном минимуме".

        - **ЕСЛИ предшествующее движение было слабым или боковым (-2% < `Контекст (30м)` < 2%):**
          Это **ИДЕАЛЬНЫЙ** сценарий. Сигнал очень сильный. **Верни "EXECUTE".**

        - Во всех остальных случаях оцени ситуацию комплексно.
        """
    
    prompt_footer = """
    **4. ТВОЙ ВЕРДИКТ (JSON):**
    `{{"action": "EXECUTE" | "REJECT", "justification": "[Твое краткое экспертное заключение]"}}`
    """

    return f"{prompt_header}\n{prompt_data}\n{prompt_task}\n{prompt_footer}".strip()



def build_golden_entry_prompt(symbol: str, side: str, reference_price: float, last_price: float, features: dict) -> str:
    """
    Создает промпт для тактического AI, который ищет подтверждение
    пробоя после фазы накопления.
    """
    price_change_pct = ((last_price - reference_price) / reference_price) * 100.0
    cvd_1m = features.get('CVD1m', 0)
    vol_anomaly = features.get('volume_anomaly', 1.0)
    
    prompt = f"""
    SYSTEM: Ты аналитик пробоев. Твоя задача — подтвердить начало истинного импульсного движения после фазы накопления. Ответ — только JSON.
    USER:
    **1. СИГНАЛ (Golden Setup - "Сжатая пружина"):**
    - Инструмент: {symbol}, Ожидаемое направление: {side.upper()}
    - Цена на момент сигнала: {reference_price:.6f}, Текущая цена: {last_price:.6f}
    - Движение от точки сигнала: {price_change_pct:+.2f}%

    **2. КОНТЕКСТ ПРОБОЯ:**
    - CVD(1m) Trend: {'Совпадает с направлением' if (side.upper() == 'BUY' and cvd_1m > 0) or (side.upper() == 'SELL' and cvd_1m < 0) else 'Противоречит'}
    - Аномалия объема (1m): {_format(vol_anomaly, '.1f')}x

    **3. ЗАДАЧА И АЛГОРИТМ:**
    Твоя задача — отличить истинный пробой от ложного.
    - **"EXECUTE":** Верни, если `Движение от точки сигнала` совпадает с направлением, `CVD Trend` совпадает, и `Аномалия объема` > 1.5x.
    - **"WAIT":** Во всех остальных случаях.

    **4. ФОРМАТ ОТВЕТА (JSON):**
    - "action": "EXECUTE" или "WAIT"
    - "reason": Краткое обоснование.
    """
    return prompt.strip()


def build_position_management_prompt(symbol: str, pos: dict, features: dict) -> str:
    """
    [V2 - РЕЖИССЕРСКИЙ] Комплексный промпт для AI-менеджера позиций.
    """
    side = pos.get('side', '').upper()
    current_roi = pos.get('current_roi', 0.0)
    
    # --- ОПРЕДЕЛЯЕМ СЦЕНАРИЙ ДЛЯ AI ---
    scenario = "PROFIT_ZONE"
    if not pos.get("trailing_activated"):
        # Если трейлинг еще не активирован
        if current_roi < 0:
            scenario = "LOSS_ZONE"
        else:
            scenario = "PRE_PROFIT_ZONE"

    prompt_header = "SYSTEM: Ты — элитный AI-менеджер позиций. Твоя задача — проанализировать ситуацию и дать комплексную рекомендацию в формате JSON."
    prompt_data = f"""
    USER:
    **1. АНАЛИЗ ОТКРЫТОЙ ПОЗИЦИИ:**
    - Инструмент: {symbol}, Направление: {side}, ROI: {current_roi:.2f}%
    - Цена входа: {pos.get('avg_price'):.6f}, Текущий стоп: {pos.get('last_stop_price', 'N/A')}
    - Контекст: RSI={_format(features.get('rsi14'), '.1f')}, ADX={_format(features.get('adx14'), '.1f')}, Тренд H1={_format(features.get('trend_h1'), '.0f')} (1=UP, -1=DOWN)
    - **СЦЕНАРИЙ:** {scenario}
    """
    
    prompt_task = ""
    if scenario == "LOSS_ZONE":
        prompt_task = """
        **2. ЗАДАЧА (Убыточная зона):**
        Позиция в убытке. Оцени, является ли это временной коррекцией в сильном тренде или началом разворота.
        - **Если тренд H1 совпадает с направлением нашей сделки:** Вероятно, это коррекция. Порекомендуй "HOLD".
        - **Если тренд H1 ПРОТИВ нашей сделки:** Риск высок. Порекомендуй подвинуть стоп (`action: "ADJUST_SL"`) ближе к цене, чтобы минимизировать убыток, если откат продолжится.
        """
    elif scenario == "PRE_PROFIT_ZONE":
        prompt_task = """
        **2. ЗАДАЧА (Зона "около нуля"):**
        Позиция долгое время не может выйти в уверенный плюс.
        - Оцени силу тренда. Если тренд слабый (ADX < 20), импульс, вероятно, иссяк. Порекомендуй установить защитный тейк-профит (`action: "SET_BREAKEVEN_TP"`) чуть выше цены входа, чтобы закрыться в небольшой плюс или безубыток.
        - Если тренд сильный, порекомендуй "HOLD".
        """
    elif scenario == "PROFIT_ZONE":
        prompt_task = """
        **2. ЗАДАЧА (Прибыльная зона):**
        Позиция в уверенном плюсе, трейлинг активирован. Твоя задача — оптимизировать выход.
        **Шаг 1: Определи силу тренда** по ADX и тренду H1.
        **Шаг 2: Рассчитай предиктивный тейк-профит.** Используй текущую цену и ATR15m (`features['atr15m']`) по формуле: `TP = Price +/- (ATR * Множитель)`. Множитель выбери в зависимости от силы тренда: 2.0 для слабого, 2.5 для среднего, 3.0 для сильного.
        **Шаг 3: Определи адекватный отступ для трейлинга.**
        - Сильный тренд (ADX > 35): отступ 3%.
        - Средний тренд (ADX 20-35): отступ 2%.
        - Слабый тренд (ADX < 20): отступ 1%.
        **Шаг 4: Верни команду `action: "SET_DYNAMIC_TP_AND_TRAIL"`** с рассчитанными `take_profit_price` и `new_trailing_gap_pct`.
        """
    
    prompt_footer = """
    **3. ФОРМАТ ОТВЕТА (JSON):**
    `{"action": "...", "reason": "...", "new_stop_price": ..., "take_profit_price": ..., "new_trailing_gap_pct": ...}`
    (возвращай только нужные для action ключи)
    """
    
    return f"{prompt_header}\n{prompt_data}\n{prompt_task}\n{prompt_footer}".strip()


def build_squeeze_entry_prompt(symbol: str, side: str, extreme_price: float, last_price: float, features: dict, funding_rate: float) -> str:
    """
    [ЭКСПЕРТНАЯ ВЕРСИЯ V5] Использует Python для всех проверок, включая фандинг,
    и просит AI принять финальное решение по "Правилу Трех Факторов".
    """
    # --- Python надежно проверяет все условия и готовит флаги ---
    
    # 1. RSI
    rsi_val = safe_to_float(features.get('rsi14'))
    rsi_ok = (side.upper() == 'SELL' and rsi_val > 75) or \
             (side.upper() == 'BUY' and rsi_val < 20)

    # 2. Фандинг
    HOT_FUNDING_THRESHOLD = 0.04
    funding_ok = (side.upper() == 'SELL' and funding_rate >= HOT_FUNDING_THRESHOLD) or \
                 (side.upper() == 'BUY' and funding_rate <= -HOT_FUNDING_THRESHOLD)

    # 3. Откат от пика/дна
    pullback_pct = 0
    if extreme_price > 0:
        pullback_pct = abs(last_price - extreme_price) / extreme_price * 100
    pullback_started = pullback_pct > 0.3

    # 4. Дивергенция по CVD
    cvd_1m = features.get('CVD1m', 0)
    cvd_divergence = (side.upper() == 'SELL' and cvd_1m < 0) or \
                     (side.upper() == 'BUY' and cvd_1m > 0)
    
    # --- Сборка промпта ---
    prompt = f"""
    SYSTEM: Ты — высокоточный тактический анализатор. Твоя задача — строго следовать алгоритму и вернуть JSON. Не рассуждай.
    USER:
    **1. СИГНАЛ НА ПОДТВЕРЖДЕНИЕ ВХОДА (Squeeze):**
    - Инструмент: {symbol}, Направление: {side.upper()}

    **2. РЕЗУЛЬТАТЫ СИСТЕМНОГО АНАЛИЗА (флаги):**
    - Благоприятный фандинг: **{funding_ok}**
    - RSI в экстремальной зоне: **{rsi_ok}**
    - Начался откат от пика/дна: **{pullback_started}**
    - CVD показывает дивергенцию: **{cvd_divergence}**

    **3. АЛГОРИТМ ПРИНЯТИЯ РЕШЕНИЯ ("Правило Трех Факторов"):**
    - **Главное условие:** `Благоприятный фандинг` должен быть `True`. Если он `False`, немедленно верни "WAIT".
    - **Если `Благоприятный фандинг` == `True`,** посчитай количество `True` среди остальных трех флагов (RSI, Откат, CVD).
    - **ЕСЛИ количество `True` >= 2:** Вернуть "EXECUTE". Это означает, что есть фандинг + как минимум два технических подтверждения.
    - **ИНАЧЕ:** Вернуть "WAIT".

    **4. ФОРМАТ ОТВЕТА (JSON):**
    - "action": "EXECUTE" или "WAIT"
    - "reason": Кратко перечисли, какие флаги были `True`.
    """
    return prompt.strip()
