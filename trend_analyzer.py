# trend_analyzer.py
import numpy as np
import pandas as pd
from ta_compat import ta
from typing import Tuple, Dict, Any


# ==============================
#   РЕЖИМЫ РЫНКА
# ==============================
REGIME_AGGR = "AGGR_START"
REGIME_SLOW = "SLOW_TREND"
REGIME_RANGE = "RANGE"


def detect_market_regime(prices: np.ndarray, volumes: np.ndarray, atr: float) -> Tuple[str, float, Dict[str, Any]]:
    """
    Анализ текущего состояния рынка: определение фазы тренда (агрессивный, медленный, боковик).
    Возвращает:
        (режим, уверенность, словарь метрик)
    """

    if len(prices) < 20:
        return REGIME_RANGE, 0.0, {}

    df = pd.DataFrame({
        "close": prices,
        "volume": volumes
    })
    df["returns"] = np.log(df["close"] / df["close"].shift(1))
    df["abs_r"] = df["returns"].abs()
    df["direction"] = np.sign(df["returns"])

    # Волатильность и ускорение
    df["volatility"] = df["abs_r"].rolling(10).mean()
    df["acceleration"] = df["volatility"].diff()

    # Наклон цены
    x = np.arange(len(df))
    coef = np.polyfit(x[-12:], df["close"].iloc[-12:], 1)
    slope = coef[0] / (atr if atr > 0 else 1)

    # Направленная доля баров
    last_sign = np.sign(df["close"].iloc[-1] - df["close"].iloc[-13])
    dir_mask = np.sign(df["returns"].iloc[-12:]) == last_sign
    dsr = dir_mask.sum() / len(dir_mask)

    # Прямолинейность (Noise-to-Signal)
    net_move = abs(df["close"].iloc[-1] - df["close"].iloc[-13])
    total_move = np.abs(df["close"].diff()).iloc[-12:].sum()
    nsr = (total_move / net_move) if net_move > 0 else 10.0

    # Объём
    vol_ratio = (df["volume"].iloc[-12:].mean() /
                 (df["volume"].iloc[-60:].mean() if len(df) > 60 else df["volume"].mean() + 1e-6))

    # "Импульс" по возрастанию волатильности
    vol_boost = (df["volatility"].iloc[-1] /
                 (df["volatility"].iloc[-15:-5].mean() + 1e-9))

    # Условия режимов
    conditions = {
        "slope": slope,
        "dsr": dsr,
        "nsr": nsr,
        "vol_boost": vol_boost,
        "vol_ratio": vol_ratio,
    }

    # Детекция
    regime = REGIME_RANGE
    score = 0

    if slope > 0.25 and dsr >= 0.65 and nsr < 1.7 and vol_boost > 1.5:
        regime = REGIME_AGGR
        score = 0.8 + min(0.2, (slope / 0.5))
    elif slope > 0.08 and dsr >= 0.55 and nsr < 2.3:
        regime = REGIME_SLOW
        score = 0.6 + min(0.2, (slope / 0.3))
    else:
        regime = REGIME_RANGE
        score = 0.3

    return regime, float(round(score, 3)), conditions


# ==============================
#   АДАПТИВНЫЙ ТРЕЙЛИНГ
# ==============================

def adaptive_trailing_stop(
    pos: Dict[str, Any],
    regime: str,
    atr: float,
    roi: float,
    last_extreme: float,
    side: str = "Buy"
) -> float:
    """
    Рассчитывает адаптивное расстояние трейлинг-стопа в зависимости от ROI, ATR и режима рынка.
    """

    if atr <= 0:
        return last_extreme

    # Базовые коэффициенты
    if regime == REGIME_AGGR:
        k_min, k_max = 0.8, 1.4
    elif regime == REGIME_SLOW:
        k_min, k_max = 1.5, 2.3
    else:  # RANGE
        k_min, k_max = 2.0, 3.0

    # Коррекция по ROI
    roi_adj = 1.0
    if roi > 0.05:
        roi_adj = max(0.7, 1.0 - (roi - 0.05) * 2.5)
    elif roi < 0:
        roi_adj = 1.1

    # Эластичный TB (trailing buffer)
    tb = atr * ((k_min + k_max) / 2) * roi_adj

    # Динамическая подстройка по ускорению (если доступна)
    acc = pos.get("acceleration", 0)
    if isinstance(acc, (int, float)):
        tb *= 1.0 - np.clip(acc, -0.5, 0.5) * 0.2

    # Защита от слишком близких стопов
    tb = np.clip(tb, 0.6 * atr, 3.0 * atr)

    if side.lower() == "buy":
        new_stop = last_extreme - tb
    else:
        new_stop = last_extreme + tb

    return float(round(new_stop, 6))


# ==============================
#   ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

def summarize_regime(regime: str, metrics: Dict[str, Any]) -> str:
    """
    Возвращает короткий текстовый отчёт для логирования.
    """
    return (
        f"{regime} | slope={metrics.get('slope', 0):.3f}, "
        f"dsr={metrics.get('dsr', 0):.2f}, "
        f"nsr={metrics.get('nsr', 0):.2f}, "
        f"vol_boost={metrics.get('vol_boost', 0):.2f}, "
        f"vol_ratio={metrics.get('vol_ratio', 0):.2f}"
    )
