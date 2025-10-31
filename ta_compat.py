"""
ta_compat
~~~~~~~~~

Обёртка поверх pandas_ta с запасным набором индикаторов.
Во время эксплуатации бота сталкивались с окружениями, где вместо pandas_ta
подсовывался облегчённый модуль ``ta_replacement`` без ADX и ATR. Это роняло
стратегии и менеджмент позиций. Модуль ниже пытается импортировать
оригинальный pandas_ta, а при отсутствии нужных функций предоставляет
минимальные fallback-реализации (EMA, SMA, RSI, ATR, ADX, Bollinger Bands,
MACD, CCI).
"""

from __future__ import annotations

import types
from typing import Any, Callable

import numpy as np
import pandas as pd

try:  # pragma: no cover - зависит от окружения
    import pandas_ta as _pta  # type: ignore
except Exception:  # pragma: no cover
    _pta = None  # type: ignore


def _to_series(values: Any, name: str | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.astype(float)
        return series.rename(name) if name else series
    return pd.Series(values, dtype="float64", name=name)


def _ewm(series: pd.Series, span: int) -> pd.Series:
    span = max(int(span), 1)
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    return series.rolling(length, min_periods=length).mean()


def _fallback_ema(series, length=10, **_: Any) -> pd.Series:
    s = _to_series(series)
    return _ewm(s, length)


def _fallback_sma(series, length=10, **_: Any) -> pd.Series:
    return _sma(_to_series(series), length)


def _fallback_rsi(series, length=14, **_: Any) -> pd.Series:
    s = _to_series(series)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _ewm(gain, length)
    avg_loss = _ewm(loss, length)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(0.0).rename(f"RSI_{length}")


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1).fillna(0.0)


def _fallback_atr(high, low, close, length=14, **_: Any) -> pd.Series:
    h = _to_series(high)
    l = _to_series(low)
    c = _to_series(close)
    tr = _true_range(h, l, c)
    atr = _ewm(tr, length)
    return atr.rename(f"ATR_{length}")


def _fallback_adx(high, low, close, length=14, **_: Any) -> pd.DataFrame:
    length = max(int(length), 1)
    h = _to_series(high)
    l = _to_series(low)
    c = _to_series(close)

    up_move = h.diff()
    down_move = -l.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0), up_move, 0.0
    )
    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0), down_move, 0.0
    )

    plus_dm = pd.Series(plus_dm, index=h.index)
    minus_dm = pd.Series(minus_dm, index=h.index)

    tr = _true_range(h, l, c)
    atr = _ewm(tr, length).replace(0.0, np.nan)

    plus_di = 100.0 * _ewm(plus_dm, length) / atr
    minus_di = 100.0 * _ewm(minus_dm, length) / atr

    di_sum = plus_di + minus_di
    dx = (plus_di - minus_di).abs() / di_sum.replace(0.0, np.nan) * 100.0
    adx = _ewm(dx.fillna(0.0), length)

    return pd.DataFrame(
        {
            f"ADX_{length}": adx.fillna(0.0),
            f"DMP_{length}": plus_di.fillna(0.0),
            f"DMN_{length}": minus_di.fillna(0.0),
        }
    )


def _fallback_bbands(series, length=20, std=2.0, **_: Any) -> pd.DataFrame:
    s = _to_series(series)
    length = max(int(length), 1)
    std = float(std)

    basis = _sma(s, length)
    dev = s.rolling(length, min_periods=length).std(ddof=0)
    upper = basis + std * dev
    lower = basis - std * dev
    bandwidth = (upper - lower).abs()
    percent = (s - lower) / bandwidth.replace(0.0, np.nan)

    suffix = f"{length}_{std:.1f}"
    return pd.DataFrame(
        {
            f"BBL_{suffix}": lower,
            f"BBM_{suffix}": basis,
            f"BBU_{suffix}": upper,
            f"BBB_{suffix}": bandwidth,
            f"BBP_{suffix}": percent,
        }
    )


def _fallback_macd(series, fast=12, slow=26, signal=9, **_: Any) -> pd.DataFrame:
    s = _to_series(series)
    fast_ema = _ewm(s, fast)
    slow_ema = _ewm(s, slow)
    macd = fast_ema - slow_ema
    signal_line = _ewm(macd, signal)
    hist = macd - signal_line
    suffix = f"{fast}_{slow}_{signal}"
    return pd.DataFrame(
        {
            f"MACD_{suffix}": macd,
            f"MACDh_{suffix}": hist,
            f"MACDs_{suffix}": signal_line,
        }
    )


def _fallback_cci(high, low, close, length=20, **_: Any) -> pd.Series:
    h = _to_series(high)
    l = _to_series(low)
    c = _to_series(close)
    tp = (h + l + c) / 3.0
    sma_tp = _sma(tp, length)
    mad = (tp - sma_tp).abs().rolling(length, min_periods=length).mean()
    cci = (tp - sma_tp) / (0.015 * mad.replace(0.0, np.nan))
    return cci.fillna(0.0).rename(f"CCI_{length}")


_FALLBACKS: dict[str, Callable[..., Any]] = {
    "ema": _fallback_ema,
    "sma": _fallback_sma,
    "rsi": _fallback_rsi,
    "atr": _fallback_atr,
    "adx": _fallback_adx,
    "bbands": _fallback_bbands,
    "macd": _fallback_macd,
    "cci": _fallback_cci,
}


class _TACompat(types.SimpleNamespace):
    def __getattr__(self, item: str) -> Any:  # pragma: no cover - тонкости импорта
        if _pta is not None and hasattr(_pta, item):
            return getattr(_pta, item)
        if item in _FALLBACKS:
            return _FALLBACKS[item]
        raise AttributeError(f"'ta' has no attribute '{item}'")


ta = _TACompat()

__all__ = ["ta"]
