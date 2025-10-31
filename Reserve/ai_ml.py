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

# def build_primary_prompt(candidate: dict, features: dict, shared_ws) -> str:
#     sym = candidate.get("symbol", "UNKNOWN")
#     side = str(candidate.get("side", "Buy")).upper()
#     source_title = str(candidate.get("source", "unknown")).replace("_", " ").title()

#     m = candidate.get("base_metrics", {})
#     vol_change_pct = safe_to_float(m.get("vol_change_pct", 0.0))
#     vol_anomaly = (vol_change_pct / 100.0) + 1.0

#     trend = "Uptrend" if safe_to_float(features.get("supertrend", 0.0)) > 0 else "Downtrend"
#     rsi_val = safe_to_float(features.get('rsi14'))
    
#     try:
#         btc_change_1h = compute_pct(shared_ws.candles_data.get("BTCUSDT", []), 60)
#         eth_change_1h = compute_pct(shared_ws.candles_data.get("ETHUSDT", []), 60)
#     except Exception:
#         btc_change_1h, eth_change_1h = 0.0, 0.0

#     prompt_header = "SYSTEM: Ты - элитный трейдер и риск-менеджер. Твой ответ - всегда только валидный JSON, без лишних слов."
    
#     prompt_data = f"""
#     USER:
#     Анализ торгового сигнала:
#     - Монета: {sym}, Направление: {side}, Источник: {source_title}
#     - Метрики: PriceΔ(5m)={_format(m.get('pct_5m'), '.2f')}%, Volume Anomaly={_format(vol_anomaly, '.1f')}x, OIΔ(1m)={_format(m.get('oi_change_pct'), '.2f')}%
#     - Контекст: Trend={trend}, ADX={_format(features.get('adx14'), '.1f')}, RSI={_format(rsi_val, '.1f')}
#     - Движение за 30м: {_format(features.get('pct_30m'), '.2f')}%
#     - Рынок: BTC Δ(1h)={_format(btc_change_1h, '.2f')}%, ETH Δ(1h)={_format(eth_change_1h, '.2f')}%
#     """
    
#     source = candidate.get("source", "")
    
#     if 'squeeze' in source:
#         rsi_condition_met = "НЕТ"
#         if side == 'SELL' and rsi_val > 75:
#             rsi_condition_met = "ДА"
#         elif side == 'BUY' and rsi_val < 25:
#             rsi_condition_met = "ДА"

#         prompt_task = f"""
#         **Стратегический контекст:** Мы торгуем КОНТРТРЕНДОВУЮ стратегию "Squeeze" (вход ПРОТИВ импульса).
#         **Правила интерпретации:**
#         1. Для входа 'SELL' (против роста) нужен перегретый RSI (>75).
#         2. Для входа 'BUY' (против падения) нужен перепроданный RSI (<25).
#         **Проверка условия RSI: {rsi_condition_met}**
#         **ЗАДАЧА:** Учитывая, выполнено ли ключевое условие по RSI, и все остальные данные, верни JSON с "action" ("EXECUTE" или "REJECT"), "confidence_score" и "justification".
#         """
#     elif 'liquidation' in source:
#         liq_side = m.get('liquidation_side', 'Unknown')
#         prompt_task = f"""
#         **Стратегический контекст:** Мы торгуем КОНТРТРЕНДОВУЮ стратегию на каскаде ликвидаций. Обнаружен крупный кластер ({liq_side}) на ${m.get('liquidation_value_usd'):,.0f}. Наша цель - войти в сделку ({side}) ПРОТИВ этих ликвидаций.
#         **Правила интерпретации:**
#         1.  Сигнал является контр-трендовым. Сильное отклонение цены и RSI являются подтверждением.
#         2.  Оцени, является ли это событие кульминацией движения (хорошо для входа) или лишь его началом (плохо).
#         **ЗАДАЧА:** Верни JSON с "action", "confidence_score" и "justification".
#         """
#     else: # Golden Setup и другие по умолчанию
#         prompt_task = """
#         **ЗАДАЧА:** Проанализируй сигнал с учетом нового критического правила. Верни JSON с ключами "confidence_score", "justification", "action" ("EXECUTE" или "REJECT").
#         **КРИТИЧЕСКОЕ ПРАВИЛО:** Наша цель — избежать входа в конце сильного импульса.
#         1.  Для сигнала 'BUY', если "Движение за 30м" уже > 5%, сигнал считается высокорискованным (покупка на пике).
#         2.  Для сигнала 'SELL', если "Движение за 30м" уже < -5%, сигнал также считается высокорискованным (продажа на дне).
#         Используй это правило как основной фактор для принятия решения о "REJECT". Если сигнал технически верен, но нарушает это правило, отклони его с соответствующим обоснованием.
#         """

#     return f"{prompt_header}\n{prompt_data}\n{prompt_task}".strip()


def build_primary_prompt(candidate: dict, features: dict, shared_ws) -> str:
    sym = candidate.get("symbol", "UNKNOWN")
    side = str(candidate.get("side", "Buy")).upper()
    source_title = str(candidate.get("source", "unknown")).replace("_", " ").title()

    m = candidate.get("base_metrics", {})
    vol_change_pct = safe_to_float(m.get("vol_change_pct", 0.0))
    vol_anomaly = (vol_change_pct / 100.0) + 1.0

    trend = "Uptrend" if safe_to_float(features.get("supertrend", 0.0)) > 0 else "Downtrend"
    rsi_val = safe_to_float(features.get('rsi14'))
    
    try:
        btc_change_1h = compute_pct(shared_ws.candles_data.get("BTCUSDT", []), 60)
        eth_change_1h = compute_pct(shared_ws.candles_data.get("ETHUSDT", []), 60)
    except Exception:
        btc_change_1h, eth_change_1h = 0.0, 0.0

    prompt_header = "SYSTEM: Ты - опытный крипто-аналитик, элитный трейдер и риск-менеджер. Твой ответ - всегда только валидный JSON, без лишних слов."
    
    prompt_data = f"""
    USER:
    Анализ торгового сигнала:
    - Монета: {sym}, Направление: {side}, Источник: {source_title}
    - Метрики: PriceΔ(5m)={_format(m.get('pct_5m'), '.2f')}%, Volume Anomaly={_format(vol_anomaly, '.1f')}x, OIΔ(1m)={_format(m.get('oi_change_pct'), '.2f')}%
    - Контекст: Trend={trend}, ADX={_format(features.get('adx14'), '.1f')}, RSI={_format(rsi_val, '.1f')}
    - Движение за 30м: {_format(features.get('pct_30m'), '.2f')}%
    - Рынок: BTC Δ(1h)={_format(btc_change_1h, '.2f')}%, ETH Δ(1h)={_format(eth_change_1h, '.2f')}%
    """
    
    source = candidate.get("source", "")
    
    if 'squeeze' in source:
        # Логика для Squeeze остается без изменений
        rsi_condition_met = "НЕТ"
        if side == 'SELL' and rsi_val > 75: rsi_condition_met = "ДА"
        elif side == 'BUY' and rsi_val < 25: rsi_condition_met = "ДА"
        prompt_task = f"""
        **Стратегический контекст:** Мы торгуем КОНТРТРЕНДОВУЮ стратегию "Squeeze", которая ищет подходящий момент на пике, или дне, движения. Желательный момент входы -- выдыхающийся импульс.
        **Правила интерпретации:** Для 'SELL' нужен RSI > 75. Для 'BUY' нужен RSI < 25.
        **Проверка условия RSI: {rsi_condition_met}**
        **ЗАДАЧА:** Учитывая все данные, верни JSON с "action", "confidence_score" и "justification".
        """
    elif 'liquidation' in source:
        # Логика для Liquidation остается без изменений
        liq_side = m.get('liquidation_side', 'Unknown')
        prompt_task = f"""
        **Стратегический контекст:** Мы торгуем КОНТРТРЕНДОВУЮ стратегию на каскаде ликвидаций в момент ликвидации максимального кластера ликвидаций.
        **ЗАДАЧА:** Оцени, является ли это событие кульминацией движения (хорошо для входа) или лишь его началом (плохо). Верни JSON с "action", "confidence_score" и "justification".
        """
    else: # Golden Setup и другие по умолчанию
        # Убираем жесткое правило, так как его выполняет Python.
        # Просим AI просто учесть контекст.
        prompt_task = """
        **ЗАДАЧА:** Проанализируй сигнал. Удели особое внимание полю "Движение за 30м". Если оно велико, это повышает риск, даже если другие метрики выглядят хорошо. Верни JSON с ключами "confidence_score", "justification", "action" ("EXECUTE" или "REJECT").
        """

    return f"{prompt_header}\n{prompt_data}\n{prompt_task}".strip()



def build_stop_management_prompt(symbol: str, pos: dict, features: dict) -> str:
    roi = safe_to_float(pos.get("pnl", 0))
    prompt = f"""
        SYSTEM: Ты - опытный крипто-трейдер и элитный риск-менеджер. Твоя задача - дать совет по управлению стоп-лоссом. Ответ - только валидный JSON.
        USER:
        **Открытая позиция:**
        - Инструмент: {symbol}, Направление: {pos.get('side', '').upper()}
        - Цена входа: {safe_to_float(pos.get('avg_price')):.6f}, Текущий ROI: {roi:.2f}%
        - Текущий трейлинг-стоп: {pos.get('last_stop_price', 'N/A')}
        **Рыночный контекст:**
        - RSI(14): {_format(features.get('rsi14'), '.1f')}, ADX(14): {_format(features.get('adx14'), '.1f')}
        - Тренд (Supertrend): {'UPTREND' if features.get('supertrend', 0) > 0 else 'DOWNTREND'}
        **ЗАДАЧА:** Дай рекомендацию в JSON.
        - "action": "MOVE_STOP" или "HOLD".
        - "new_stop_price": 123.45 (только если action="MOVE_STOP").
        - "reason": Краткое обоснование.
        """
    return prompt.strip()

def build_squeeze_entry_prompt(symbol: str, side: str, extreme_price: float, last_price: float, features: dict) -> str:
    pullback_pct = abs(last_price - extreme_price) / extreme_price * 100
    cvd_1m = features.get('CVD1m', 0)
    vol_1m = features.get('vol1m', 0)
    avg_vol_30m = features.get('avgVol30m', 1)
    vol_anomaly = vol_1m / avg_vol_30m if avg_vol_30m > 0 else 1.0

    prompt = f"""
    SYSTEM: Ты аналитик микроструктуры рынка. Твоя задача - определить оптимальный момент для КОНТРТРЕНДОВОГО входа в момент выдыхающегося сквиза. Ответ - только валидный JSON.
    USER:
    **Сигнал (Squeeze):**
    - Инструмент: {symbol}, Направление входа: {side.upper()}
    - Пик/дно импульса: {extreme_price:.6f}, Текущая цена: {last_price:.6f}
    - Откат от пика/дна: {pullback_pct:.2f}%
    **Контекст отката:**
    - RSI(14): {_format(features.get('rsi14'), '.1f')}
    - CVD(1m) Trend: {'Растет' if cvd_1m > 0 else 'Падает'} ({cvd_1m:,.0f})
    - Аномалия объема (1m): {_format(vol_anomaly, '.1f')}x от среднего
    **ЗАДАЧА:** Ищи признаки ИСТОЩЕНИЯ импульса. Является ли СЕЙЧАС оптимальный момент для входа? Верни JSON:
    - "action": "EXECUTE" (только при явных признаках разворота) или "WAIT".
    - "reason": Краткое обоснование.
    """
    return prompt.strip()