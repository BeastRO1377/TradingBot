import asyncio
import logging
import time
import math
import json
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timezone
from pathlib import Path
from collections import deque, defaultdict
from decimal import Decimal
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from coremltools.models import MLModel

from pybit.unified_trading import HTTP, WebSocket
from ratelimiter import AsyncLimiter
from pybit.exceptions import InvalidRequestError
from models import GoldenNet, train_model

# ————————————————————————————————————————————————————————————————
# Конфигурационные параметры

logger = logging.getLogger("bot")

# Глобальные параметры (все основные настройки приходят из user_state.json)
SQUEEZE_COOLDOWN_SEC = 600
DEFAULT_TRAILING_START_PCT = 0.4
DEFAULT_TRAILING_GAP_PCT = 0.3
DEC_TICK = 0.0001
VOLUME_COEF = 1.2
DEVICE = torch.device("mps")

# ————————————————————————————————————————————————————————————————
# Вспомогательные функции

def safe_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def sum_last_vol(candles, minutes=5):
    sum_vol = 0.0
    for bar in candles[-minutes:]:
        vol = safe_to_float(bar.get("volume") or bar.get("turnover", 0))
        sum_vol += vol
    return sum_vol

def _append_snapshot(row: dict):
    snapshot_file = Path("golden_snapshots.json")
    data = []
    if snapshot_file.exists():
        try:
            with open(snapshot_file, "r") as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(row)
    with open(snapshot_file, "w") as f:
        json.dump(data, f, indent=2)

# ————————————————————————————————————————————————————————————————
# Основной ML-инфер

class MLInferencer:
    def __init__(self, model_path="model.txt"):
        self.model = None
        self.feature_scaler = StandardScaler()
        self.FEATURE_KEYS = [
            "price", "pct1m", "pct5m", "pct15m",
            "vol1m", "vol5m", "vol15m",
            "OI_now", "dOI1m", "dOI5m",
            "spread_pct", "sigma5m", "CVD1m", "CVD5m",
            "rsi14", "sma50", "ema20", "atr14", "bb_width",
            "supertrend", "cci20", "macd", "macd_signal",
            "avgVol30m", "avgOI30m", "deltaCVD30m",
            "GS_pct4m", "GS_vol4m", "GS_dOI4m", "GS_cvd4m",
            "GS_supertrend", "GS_cooldown",
            "SQ_pct1m", "SQ_pct5m", "SQ_vol1m", "SQ_vol5m", "SQ_dOI1m",
            "SQ_spread_pct", "SQ_sigma5m", "SQ_liq10s", "SQ_cooldown",
            "LIQ_cluster_val10s", "LIQ_cluster_count10s", "LIQ_direction",
            "LIQ_pct1m", "LIQ_pct5m", "LIQ_vol1m", "LIQ_vol5m", "LIQ_dOI1m",
            "LIQ_spread_pct", "LIQ_sigma5m", "LIQ_golden_flag", "LIQ_squeeze_flag",
            "LIQ_cooldown",
            "hour_of_day", "day_of_week", "month_of_year", "adx14",
        ]

        try:
            self.model = LGBMClassifier()
            self.model.booster_ = self.model.booster_.load_model(model_path)
            logger.info("[ML] LightGBM model loaded successfully")
        except Exception as e:
            logger.warning(f"[ML] Model load failed: {e}")

    def predict(self, X):
        if self.model:
            return self.model.predict(X)
        return None
    
    # ————————————————————————————————————————————————————————————————
# Модель GoldenNet (PyTorch версия, оптимизированная для NE/CoreML)

import torch
import torch.nn as nn

class GoldenNet(nn.Module):
    def __init__(self, input_size=65):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 2)  # 2 класса: вход или не вход

        self.dropout = nn.Dropout(0.1)
        self.act = nn.SiLU()  # или nn.GELU()

    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.act(self.bn2(self.fc2(x)))
        x = self.act(self.fc3(x))
        x = self.dropout(x)
        out = self.fc_out(x)
        return out
    
    # ————————————————————————————————————————————————————————————————
# Буфер данных трейдов для обучения (подготовка данных под Trainer)

from collections import deque
import numpy as np

class TradeBuffer:
    """
    Буфер для хранения трейдов и подготовки данных для обучения.
    """

    def __init__(self, maxlen=50000):
        self.buffer = deque(maxlen=maxlen)

    def append(self, features, label):
        """
        features: numpy.array (shape: [n_features])
        label: int (0 или 1)
        """
        self.buffer.append((features, label))

    def get_batch(self, batch_size=1024):
        """
        Получить случайную мини-выборку для тренировки
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch_features = []
        batch_labels = []
        for idx in indices:
            features, label = self.buffer[idx]
            batch_features.append(features)
            batch_labels.append(label)
        return np.array(batch_features, dtype=np.float32), np.array(batch_labels, dtype=np.int64)

    def __len__(self):
        return len(self.buffer)
    
    # ————————————————————————————————————————————————————————————————
# Финальный Trainer для GoldenNet V18.6 (инкрементальное обучение)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

class GoldenTrainer:
    def __init__(self, model: GoldenNet, buffer: TradeBuffer, lr=1e-4, batch_size=1024, device=None):
        self.model = model
        self.buffer = buffer
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train_one_epoch(self):
        """
        Обучение на одном проходе по буферу
        """
        if len(self.buffer) < self.batch_size:
            print("[Trainer] Недостаточно данных в буфере для обучения")
            return

        X_batch, y_batch = self.buffer.get_batch(self.batch_size)
        X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_batch, dtype=torch.float32).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(X_tensor).squeeze()
        loss = self.criterion(outputs, y_tensor)
        loss.backward()
        self.optimizer.step()

        print(f"[Trainer] Loss: {loss.item():.6f}")

    def train_loop(self, epochs=1):
        """
        Несколько эпох подряд.
        """
        for epoch in range(epochs):
            self.train_one_epoch()

    def save_model(self, path="golden_model_v18_6.pt"):
        torch.save(self.model.state_dict(), path)
        print(f"[Trainer] Модель сохранена: {path}")

    def load_model(self, path="golden_model_v18_6.pt"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"[Trainer] Модель загружена: {path}")

# ————————————————————————————————————————————————————————————————
# МОДУЛЬ ИНТЕГРАЦИИ GoldenNet В TradingBot (LIVE INFERENCE)

class MLInferencer:
    """
    Инференс-модуль для TradingBot с GoldenNet V18.6
    """
    def __init__(self, model_path: str, feature_keys: list, scaler_path: str = None, device=None):
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Подгружаем модель
        self.model = GoldenNet(input_size=len(feature_keys)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Загружаем scaler (если есть)
        self.scaler = None
        if scaler_path:
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        self.feature_keys = feature_keys

    def predict_entry(self, features: dict) -> float:
        """
        features — dict (feature_name → значение)
        """
        # Собираем в numpy массив
        X = np.array([features.get(k, 0.0) for k in self.feature_keys], dtype=np.float32).reshape(1, -1)

        # Масштабируем
        if self.scaler:
            X = self.scaler.transform(X)

        # Переводим в Tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self.model(X_tensor).cpu().numpy().squeeze()

        return output  # вероятность (0.0 — нет входа; 1.0 — сильный сигнал)

    def predict_binary(self, features: dict, threshold=0.5) -> bool:
        score = self.predict_entry(features)
        return score >= threshold
# ————————————————————————————————————————————————————————————————————
# ИНТЕГРАЦИЯ MLInferencer В TradingBot

class TradingBot:
    def __init__(self, user_data, shared_ws, golden_param_store):
        # ... твоя большая инициализация (уже внедрённая ранее)
        self.device = DEVICE

        # ML-инференс инициализация:
        model_path = user_data.get("ml_model_path", "golden_v18.pth")
        scaler_path = user_data.get("ml_scaler_path", "golden_scaler.pkl")
        self.feature_keys = [
            "price", "pct1m", "pct5m", "pct15m", "vol1m", "vol5m", "vol15m",
            "OI_now", "dOI1m", "dOI5m", "spread_pct", "sigma5m", "CVD1m",
            "CVD5m", "rsi14", "sma50", "ema20", "atr14", "bb_width",
            "supertrend", "cci20", "macd", "macd_signal", "avgVol30m",
            "avgOI30m", "deltaCVD30m", "GS_pct4m", "GS_vol4m", "GS_dOI4m",
            "GS_cvd4m", "GS_supertrend", "GS_cooldown", "SQ_pct1m", "SQ_pct5m",
            "SQ_vol1m", "SQ_vol5m", "SQ_dOI1m", "SQ_spread_pct", "SQ_sigma5m",
            "SQ_liq10s", "SQ_cooldown", "LIQ_cluster_val10s", "LIQ_cluster_count10s",
            "LIQ_direction", "LIQ_pct1m", "LIQ_pct5m", "LIQ_vol1m", "LIQ_vol5m",
            "LIQ_dOI1m", "LIQ_spread_pct", "LIQ_sigma5m", "LIQ_golden_flag",
            "LIQ_squeeze_flag", "LIQ_cooldown", "hour_of_day", "day_of_week",
            "month_of_year", "adx14"
        ]

        try:
            self.inferencer = MLInferencer(model_path, self.feature_keys, scaler_path, device=self.device)
            logger.info("[ML] Инференс-модуль GoldenNet V18.6 успешно инициализирован.")
        except Exception as e:
            logger.warning(f"[ML] Ошибка при загрузке модели: {e}")
            self.inferencer = None