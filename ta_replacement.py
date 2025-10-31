# ta_replacement.py - Замена для pandas_ta функций
import pandas as pd
import numpy as np

def atr(high, low, close, length=14):
    """
    Average True Range (ATR) - простая реализация
    """
    if len(high) < length + 1:
        return pd.Series([np.nan] * len(high), index=high.index)
    
    # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR = скользящее среднее True Range
    atr_values = true_range.rolling(window=length).mean()
    
    return atr_values

def rsi(close, length=14):
    """
    Relative Strength Index (RSI) - простая реализация
    """
    if len(close) < length + 1:
        return pd.Series([np.nan] * len(close), index=close.index)
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    
    rs = gain / loss
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values

def sma(close, length=20):
    """
    Simple Moving Average (SMA)
    """
    return close.rolling(window=length).mean()

def ema(close, length=20):
    """
    Exponential Moving Average (EMA)
    """
    return close.ewm(span=length).mean()

def bbands(close, length=20, std=2):
    """
    Bollinger Bands - простая реализация
    """
    sma_values = sma(close, length)
    std_values = close.rolling(window=length).std()
    
    upper_band = sma_values + (std_values * std)
    middle_band = sma_values
    lower_band = sma_values - (std_values * std)
    
    return pd.DataFrame({
        'BBU': upper_band,
        'BBM': middle_band,
        'BBL': lower_band
    })

















