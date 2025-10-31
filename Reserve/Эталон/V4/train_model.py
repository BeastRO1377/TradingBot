# train_model.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# --- НАЧАЛО ИСПРАВЛЕНИЯ: Monkey patch для numpy 2.0 ---
# Эта строка исправляет ошибку ImportError: cannot import name 'NaN' from 'numpy'
# Она должна быть до импорта config и ai_ml, которые импортируют pandas-ta.
if not hasattr(np, "NaN"):
    np.NaN = np.nan
# --- КОНЕЦ ИСПРАВЛЕНИЯ ---

import config
from ai_ml import train_golden_model_mlx, save_mlx_checkpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_training_data(csv_path: Path) -> list:
    """
    Загружает данные из CSV, готовит их к обучению, используя все доступные сделки.
    """
    if not csv_path.exists():
        logging.error(f"Файл с данными не найден: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    logging.info(f"Загружено {len(df)} записей из {csv_path.name}")

    # 1. Используем только события открытия позиций
    df_filtered = df[df['event'] == 'open'].copy()

    # 2. Создаем "целевую переменную" (target)
    close_events = df[df['event'] == 'close']
    
    targets = []
    for _, open_row in df_filtered.iterrows():
        corresponding_close = close_events[
            (close_events['symbol'] == open_row['symbol']) &
            (close_events['timestamp'] > open_row['timestamp'])
        ].sort_values('timestamp').iloc[:1]
        
        if not corresponding_close.empty:
            pnl_pct = corresponding_close['pnl_pct'].values[0]
            targets.append(1 if pnl_pct > 0.5 else 0)
        else:
            targets.append(np.nan)

    df_filtered['target'] = targets
    
    df_filtered.dropna(subset=['target'], inplace=True)
    logging.info(f"Найдено {len(df_filtered)} полных сделок (открытие+закрытие) для обучения.")

    training_data = []
    for _, row in df_filtered.iterrows():
        # Собираем вектор признаков в соответствии с config.py
        features = [
            row['price'],
            row['open_interest'],
            row['volume_1m'],
            row['rsi14'],
            row['adx14'],
            row['volume_anomaly']
        ]
        
        if not all(isinstance(f, (int, float)) and np.isfinite(f) for f in features):
            continue

        training_data.append({
            "features": features,
            "target": row['target']
        })
        
    return training_data

# --- НАЧАЛО ИСПРАВЛЕНИЯ: Правильный синтаксис для __main__ ---
if __name__ == "__main__":
    logging.info("Начинаю процесс обучения модели...")
    
    prepared_data = prepare_training_data(config.TRADES_UNIFIED_CSV_PATH)
    
    if len(prepared_data) < 50:
        logging.error(f"Недостаточно данных для обучения ({len(prepared_data)} сэмплов). Процесс прерван.")
    else:
        model, scaler = train_golden_model_mlx(prepared_data, num_epochs=50)
        save_mlx_checkpoint(model, scaler)
        logging.info("Обучение и сохранение модели успешно завершено!")
# --- КОНЕЦ ИСПРАВЛЕНИЯ ---