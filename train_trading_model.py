#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для обучения MLX-модели на основе истории торгов из RESULTS.csv.
Он извлекает логику из основного файла бота, подготавливает данные и
сохраняет обученную модель и скейлер для использования в реальной торговле.
"""

import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
from collections import deque, defaultdict
import json
import logging
import time

# ======================================================================
# Код для совместимости с NumPy 2.0+
# ======================================================================
if not hasattr(np, "NaN"):
    np.NaN = np.nan
# ======================================================================

# --- Библиотеки для ML ---
import mlx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim
from sklearn.preprocessing import StandardScaler
from safetensors.numpy import save_file as save_safetensors
import joblib
import pandas_ta as ta
import warnings
from sklearn.exceptions import InconsistentVersionWarning


# ======================================================================
# СКОПИРОВАННЫЙ И АДАПТИРОВАННЫЙ КОД ИЗ MultiuserBot_v2RC_separated.py
# ======================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

FEATURE_KEYS = [
    "price", "pct1m", "pct5m", "pct15m", "vol1m", "vol5m", "vol15m",
    "OI_now", "dOI1m", "dOI5m", "spread_pct", "sigma5m", "CVD1m", "CVD5m",
    "rsi14", "sma50", "ema20", "atr14", "bb_width", "supertrend", "cci20",
    "macd", "macd_signal", "avgVol30m", "avgOI30m", "deltaCVD30m", "GS_pct4m",
    "GS_vol4m", "GS_dOI4m", "GS_cvd4m", "GS_supertrend", "GS_cooldown",
    "SQ_pct1m", "SQ_pct5m", "SQ_vol1m", "SQ_vol5m", "SQ_dOI1m",
    "SQ_spread_pct", "SQ_sigma5m", "SQ_liq10s", "SQ_power", "SQ_strength",
    "SQ_cooldown", "LIQ_cluster_val10s", "LIQ_cluster_count10s",
    "LIQ_direction", "LIQ_pct1m", "LIQ_pct5m", "LIQ_vol1m", "LIQ_vol5m",
    "LIQ_dOI1m", "LIQ_spread_pct", "LIQ_sigma5m", "LIQ_golden_flag",
    "LIQ_squeeze_flag", "LIQ_cooldown", "hour_of_day", "day_of_week",
    "month_of_year", "adx14",
]
INPUT_DIM = len(FEATURE_KEYS)
VOL_WINDOW = 60

def safe_to_float(val, default=0.0):
    try:
        if isinstance(val, str): val = val.replace(',', '.')
        return float(val)
    except (ValueError, TypeError, AttributeError):
        return default

def compute_pct(candles_deque, minutes: int) -> float:
    data = list(candles_deque)
    if len(data) < minutes + 1: return 0.0
    old_close = safe_to_float(data[-minutes - 1].get("closePrice", 0))
    new_close = safe_to_float(data[-1].get("closePrice", 0))
    if old_close <= 0: return 0.0
    return (new_close - old_close) / old_close * 100.0

def sum_last_vol(candles_deque, minutes: int) -> float:
    data = list(candles_deque)[-minutes:]
    return sum(safe_to_float(c.get("volume", 0)) for c in data)

def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3):
    if len(df) < (period + 1) * 2: return pd.Series([False] * len(df), index=df.index, dtype=bool)
    high, low, close = df["highPrice"].astype("float32"), df["lowPrice"].astype("float32"), df["closePrice"].astype("float32")
    atr = ta.atr(high, low, close, length=period)
    if atr is None or atr.isna().all(): return pd.Series([False] * len(df), index=df.index, dtype=bool)
    hl2 = (high + low) / 2
    upperband, lowerband = hl2 + multiplier * atr, hl2 - multiplier * atr
    supertrend = pd.Series(index=df.index, dtype=bool)
    in_uptrend = True
    for i in range(len(df)):
        if i == 0:
            supertrend.iat[i] = in_uptrend
            continue
        if close.iat[i] > upperband.iat[i - 1]: in_uptrend = True
        elif close.iat[i] < lowerband.iat[i - 1]: in_uptrend = False
        if in_uptrend and lowerband.iat[i] < lowerband.iat[i - 1]: lowerband.iat[i] = lowerband.iat[i - 1]
        if not in_uptrend and upperband.iat[i] > upperband.iat[i - 1]: upperband.iat[i] = upperband.iat[i - 1]
        supertrend.iat[i] = in_uptrend
    return supertrend

def _sigma_5m(candles_deque, window: int = VOL_WINDOW) -> float:
    candles = list(candles_deque)[-window:]
    if len(candles) < window: return 0.0
    prices = [c.get("openPrice") for c in candles]
    if not any(p is not None and p > 0 for p in prices): return 0.0
    moves = [abs(c["closePrice"] - c["openPrice"]) / c["openPrice"] for c in candles if c.get("openPrice") > 0]
    return float(np.std(moves)) if moves else 0.0

class MockDataManager:
    def __init__(self):
        self.candles_data = defaultdict(lambda: deque(maxlen=1000))
        self.oi_history = defaultdict(lambda: deque(maxlen=1000))
        self.cvd_history = defaultdict(lambda: deque(maxlen=1000))
        self.latest_open_interest, self.ticker_data = {}, {}
    def _sigma_5m(self, symbol: str) -> float: return _sigma_5m(self.candles_data.get(symbol, []), VOL_WINDOW)
    def _golden_allowed(self, symbol): return True
    def _squeeze_allowed(self, symbol): return True
    def check_liq_cooldown(self, symbol): return True

def extract_realtime_features(symbol: str, data_manager: MockDataManager) -> dict:
    candles_deque = data_manager.candles_data.get(symbol, deque())
    if not candles_deque: return {}
    last_candle = candles_deque[-1]
    last_price = safe_to_float(last_candle.get("closePrice", 0.0))
    if last_price <= 0.0: return {}
    bid1, ask1 = last_price * 0.9999, last_price * 1.0001
    spread_pct = (ask1 - bid1) / bid1 * 100.0 if bid1 > 0 else 0.0
    oi_hist, cvd_hist = list(data_manager.oi_history.get(symbol, [])), list(data_manager.cvd_history.get(symbol, []))
    pct1m, pct5m, pct15m = compute_pct(candles_deque, 1), compute_pct(candles_deque, 5), compute_pct(candles_deque, 15)
    V1m, V5m, V15m = sum_last_vol(candles_deque, 1), sum_last_vol(candles_deque, 5), sum_last_vol(candles_deque, 15)
    OI_now, OI_prev1m, OI_prev5m = (safe_to_float(oi_hist[-1]) if oi_hist else 0.0), (safe_to_float(oi_hist[-2]) if len(oi_hist) >= 2 else 0.0), (safe_to_float(oi_hist[-6]) if len(oi_hist) >= 6 else 0.0)
    dOI1m = (OI_now - OI_prev1m) / OI_prev1m if OI_prev1m > 0 else 0.0
    dOI5m = (OI_now - OI_prev5m) / OI_prev5m if OI_prev5m > 0 else 0.0
    CVD_now, CVD_prev1m, CVD_prev5m = (safe_to_float(cvd_hist[-1]) if cvd_hist else 0.0), (safe_to_float(cvd_hist[-2]) if len(cvd_hist) >= 2 else 0.0), (safe_to_float(cvd_hist[-6]) if len(cvd_hist) >= 6 else 0.0)
    CVD1m, CVD5m = CVD_now - CVD_prev1m, CVD_now - CVD_prev5m
    sigma5m = data_manager._sigma_5m(symbol)
    df = pd.DataFrame(list(candles_deque)[-100:])
    for col in ("openPrice", "closePrice", "highPrice", "lowPrice", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill().bfill()
    n = len(df)
    close, high, low = df["closePrice"], df["highPrice"], df["lowPrice"]
    def _safe_last(s, default=0.0): return float(s.iloc[-1]) if not s.empty and pd.notna(s.iloc[-1]) else default
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rsi14 = _safe_last(ta.rsi(close, length=14), 50.0) if n >= 15 else 50.0
        sma50 = _safe_last(ta.sma(close, length=50), _safe_last(close, 0.0)) if n >= 50 else _safe_last(close, 0.0)
        ema20 = _safe_last(ta.ema(close, length=20), sma50) if n >= 20 else sma50
        atr14 = _safe_last(ta.atr(high, low, close, length=14), 0.0) if n >= 15 else 0.0
        bbands = ta.bbands(close, length=20)
        bb_width = _safe_last(bbands.iloc[:, 2] - bbands.iloc[:, 0], 0.0) if bbands is not None and n >= 20 else 0.0
        supertrend_val = _safe_last(compute_supertrend(df), 0.0) if n > 20 else 0.0
        adx_df = ta.adx(high, low, close, length=14)
        adx14 = _safe_last(adx_df["ADX_14"], 0.0) if adx_df is not None and n >= 15 else 0.0
        cci20 = _safe_last(ta.cci(high, low, close, length=20), 0.0) if n >= 20 else 0.0
        macd_df = ta.macd(close, fast=12, slow=26, signal=9) if n >= 35 else None
    macd_val = _safe_last(macd_df.iloc[:, 0], 0.0) if macd_df is not None else 0.0
    macd_signal = _safe_last(macd_df.iloc[:, 2], 0.0) if macd_df is not None else 0.0
    avgVol30m = np.mean([c['volume'] for c in list(candles_deque)[-30:]]) if len(candles_deque) >= 30 else 0.0
    avgOI30m = np.mean(oi_hist[-30:]) if len(oi_hist) >= 30 else 0.0
    deltaCVD30m = CVD_now - (safe_to_float(cvd_hist[-31]) if len(cvd_hist) >= 31 else 0.0)
    features = {k: 0.0 for k in FEATURE_KEYS}
    features.update({
        "price": last_price, "pct1m": pct1m, "pct5m": pct5m, "pct15m": pct15m,
        "vol1m": V1m, "vol5m": V5m, "vol15m": V15m, "OI_now": OI_now, "dOI1m": dOI1m, "dOI5m": dOI5m,
        "spread_pct": spread_pct, "sigma5m": sigma5m, "CVD1m": CVD1m, "CVD5m": CVD5m,
        "rsi14": rsi14, "sma50": sma50, "ema20": ema20, "atr14": atr14, "bb_width": bb_width,
        "supertrend": 1 if supertrend_val else -1, "cci20": cci20, "macd": macd_val, "macd_signal": macd_signal,
        "avgVol30m": avgVol30m, "avgOI30m": avgOI30m, "deltaCVD30m": deltaCVD30m, "adx14": adx14,
        "hour_of_day": dt.datetime.now().hour, "day_of_week": dt.datetime.now().weekday(), "month_of_year": dt.datetime.now().month,
    })
    return features

class GoldenNetMLX(mlx_nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1, self.bn1 = mlx_nn.Linear(input_size, hidden_size), mlx_nn.BatchNorm(hidden_size)
        self.fc2, self.bn2 = mlx_nn.Linear(hidden_size, hidden_size), mlx_nn.BatchNorm(hidden_size)
        self.fc3, self.dropout = mlx_nn.Linear(hidden_size, 1), mlx_nn.Dropout(0.2)
    def __call__(self, x):
        x = mlx_nn.relu(self.bn1(self.fc1(x))); x = self.dropout(x)
        x = mlx_nn.relu(self.bn2(self.fc2(x))); x = self.fc3(x)
        return x
    def state_dict_numpy(self) -> dict:
        to_np = lambda t: np.array(t)
        return {"fc1.weight": to_np(self.fc1.weight), "fc1.bias": to_np(self.fc1.bias),
                "bn1.weight": to_np(self.bn1.weight), "bn1.bias": to_np(self.bn1.bias),
                "bn1.running_mean": to_np(self.bn1.running_mean), "bn1.running_var": to_np(self.bn1.running_var),
                "fc2.weight": to_np(self.fc2.weight), "fc2.bias": to_np(self.fc2.bias),
                "bn2.weight": to_np(self.bn2.weight), "bn2.bias": to_np(self.bn2.bias),
                "bn2.running_mean": to_np(self.bn2.running_mean), "bn2.running_var": to_np(self.bn2.running_var),
                "fc3.weight": to_np(self.fc3.weight), "fc3.bias": to_np(self.fc3.bias)}

def train_golden_model_mlx(training_data, num_epochs: int = 30, lr: float = 1e-3):
    logger.info("[MLX] Запуск обучения на MLX...")
    feats = np.asarray([d["features"] for d in training_data], dtype=np.float32)
    targ = np.asarray([d["target"] for d in training_data], dtype=np.float32)
    mask = ~(np.isnan(feats).any(1) | np.isinf(feats).any(1) | np.isnan(targ) | np.isinf(targ))
    feats, targ = feats[mask], targ[mask]
    if feats.size == 0: raise ValueError("train_golden_model_mlx: нет валидных сэмплов")
    scaler = StandardScaler().fit(feats)
    feats_scaled = scaler.transform(feats).astype(np.float32)
    targ = targ.reshape(-1, 1)
    model = GoldenNetMLX(input_size=feats_scaled.shape[1])
    optimizer = mlx_optim.Adam(learning_rate=lr)
    loss_fn = lambda model, x, y: mlx_nn.losses.mse_loss(model(x), y).mean()
    loss_and_grad_fn = mlx_nn.value_and_grad(model, loss_fn)
    for epoch in range(num_epochs):
        x_train, y_train = mlx.core.array(feats_scaled), mlx.core.array(targ)
        loss, grads = loss_and_grad_fn(model, x_train, y_train)
        optimizer.update(model, grads)
        mlx.core.eval(model.parameters(), optimizer.state)
        if (epoch + 1) % 5 == 0: logger.info(f"Epoch {epoch+1} [MLX] – Loss: {loss.item():.5f}")
    return model, scaler

def save_mlx_checkpoint(model: GoldenNetMLX, scaler: StandardScaler,
                        model_path: str = "golden_model_mlx.safetensors",
                        scaler_path: str = "scaler.pkl"):
    tensors = model.state_dict_numpy()
    save_safetensors(tensors, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info("[MLX] Модель сохранена → %s; scaler → %s", model_path, scaler_path)

def process_trades_and_create_dataset(df: pd.DataFrame):
    logger.info("Начинаем обработку сделок для создания датасета...")
    data_manager, training_samples = MockDataManager(), []
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df['minute'] = df['timestamp'].dt.floor('min')
    
    all_symbols = df['symbol'].unique()
    
    for symbol in all_symbols:
        df_sym = df[df['symbol'] == symbol].copy()
        if df_sym.empty: continue
            
        logger.info(f"Обработка {symbol}...")
        
        full_time_range = pd.date_range(start=df_sym['minute'].min(), end=df_sym['minute'].max(), freq='min')
        df_resampled = df_sym.set_index('timestamp').resample('min').agg({
            'price': ['first', 'max', 'min', 'last'], 'volume': 'sum'
        }).reindex(full_time_range)
        
        df_resampled.columns = ['open', 'high', 'low', 'close', 'volume']
        df_resampled[['open', 'high', 'low', 'close']] = df_resampled[['open', 'high', 'low', 'close']].interpolate(method='time')
        df_resampled['volume'] = df_resampled['volume'].fillna(0)
        df_resampled = df_resampled.reset_index().rename(columns={'index': 'minute'})
        
        data_manager.candles_data[symbol].clear()
        data_manager.oi_history[symbol].clear()
        data_manager.cvd_history[symbol].clear()
        
        open_trade = None
        for index, row in df_sym.iterrows():
            current_minute = row['minute']
            bars_to_process = df_resampled[df_resampled['minute'] <= current_minute]
            last_processed_ts = data_manager.candles_data[symbol][-1]['startTime'] if data_manager.candles_data[symbol] else pd.Timestamp(0)
            new_bars = bars_to_process[bars_to_process['minute'] > last_processed_ts]
            
            for _, bar in new_bars.iterrows():
                candle_data = {'startTime': bar['minute'], 'openPrice': bar['open'], 'highPrice': bar['high'], 'lowPrice': bar['low'], 'closePrice': bar['close'], 'volume': bar['volume']}
                data_manager.candles_data[symbol].append(candle_data)
                data_manager.oi_history[symbol].append(0)
                delta = bar['volume'] if bar['close'] >= bar['open'] else -bar['volume']
                prev_cvd = data_manager.cvd_history[symbol][-1] if data_manager.cvd_history[symbol] else 0
                data_manager.cvd_history[symbol].append(prev_cvd + delta)

            # ИСПРАВЛЕННАЯ ЛОГИКА: Новый 'open' всегда начинает новую сделку
            if row['event'] == 'open' and row['result'] == 'opened':
                if len(data_manager.candles_data[symbol]) >= 50:
                    features = extract_realtime_features(symbol, data_manager)
                    if features:
                        if open_trade is not None:
                            logger.debug(f"Обнаружен новый 'open' для {symbol} до закрытия предыдущего. Старая точка входа отброшена.")
                        open_trade = {'features': list(features.values())}

            elif row['event'] == 'close' and open_trade is not None:
                pnl_pct = safe_to_float(row['result.1'], default=None)
                if pnl_pct is not None:
                    training_samples.append({'features': open_trade['features'], 'target': pnl_pct})
                open_trade = None

    logger.info(f"Обработка завершена. Сформировано {len(training_samples)} обучающих примеров.")
    return training_samples

if __name__ == "__main__":
    csv_path = "RESULTS.csv"
    if not os.path.exists(csv_path):
        logger.critical(f"Файл {csv_path} не найден!")
        sys.exit(1)
    try:
        df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip', engine='python')
        logger.info(f"Загружен файл {csv_path}, строк: {len(df)}")
        df.columns = [c.strip() for c in df.columns]
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna(subset=['timestamp', 'symbol', 'event', 'result'])
        for col in ['volume', 'price', 'result', 'result.1']:
            if col in df.columns:
                if df[col].dtype == 'object':
                    replacements = {'янв': '1', 'фев': '2', 'март': '3', 'апр': '4', 'май': '5', 'июнь': '6', 'июль': '7', 'авг': '8', 'сент': '9', 'окт': '10', 'нояб': '11', 'дек': '12'}
                    for k, v in replacements.items():
                        df[col] = df[col].str.replace(str(k), str(v), case=False, regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception as e:
        logger.critical(f"Ошибка при чтении или обработке CSV: {e}", exc_info=True)
        sys.exit(1)
    
    training_data = process_trades_and_create_dataset(df)
    
    if not training_data or len(training_data) < 50:
        logger.critical(f"Не удалось сформировать достаточное количество обучающих примеров. Найдено всего {len(training_data)}.")
        sys.exit(1)
        
    try:
        model, scaler = train_golden_model_mlx(training_data, num_epochs=50) 
    except Exception as e:
        logger.critical(f"Критическая ошибка во время обучения модели: {e}", exc_info=True)
        sys.exit(1)
        
    try:
        save_mlx_checkpoint(model, scaler)
        logger.info("=" * 50)
        logger.info("      Обучение успешно завершено!                  ")
        logger.info("      Файлы golden_model_mlx.safetensors и scaler.pkl созданы.")
        logger.info("=" * 50)
    except Exception as e:
        logger.critical(f"Критическая ошибка при сохранении модели: {e}", exc_info=True)
        sys.exit(1)