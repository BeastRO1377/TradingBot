# retrain_and_validate.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import shutil
import time  # <--- ДОБАВЛЕН ИСПРАВЛЯЮЩИЙ ИМПОРТ
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Устанавливаем numpy.NaN, если атрибут отсутствует (для совместимости)
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import config
from ai_ml import train_golden_model_mlx, save_mlx_checkpoint, MLXInferencer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_training_data(csv_path: Path) -> list:
    """Загружает и готовит данные из trades_unified.csv."""
    if not csv_path.exists():
        logging.error(f"Файл с данными не найден: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    df_filtered = df[df['event'] == 'open'].copy()
    
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
    
    training_data = []
    # --- ИЗМЕНЕНИЕ: Убедимся, что ключи фичей соответствуют последней версии модели ---
    feature_keys = [
        "price", "open_interest", "volume_1m", "rsi14", "adx14", "volume_anomaly"
    ]
    for _, row in df_filtered.iterrows():
        # Проверяем наличие всех ключей, чтобы избежать KeyError
        if not all(key in row for key in feature_keys):
            continue
            
        features = [row[k] for k in feature_keys]
        if not all(isinstance(f, (int, float)) and np.isfinite(f) for f in features):
            continue
        training_data.append({"features": features, "target": row['target']})
        
    return training_data

def evaluate_model(inferencer: MLXInferencer, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Оценивает производительность модели на тестовых данных."""
    if inferencer.model is None or X_test.size == 0:
        return {"accuracy": 0, "f1": 0, "precision": 0, "recall": 0}

    # Получаем предсказания
    raw_predictions = inferencer.infer(X_test)
    # Применяем сигмоиду и порог 0.5 для бинарной классификации
    probabilities = 1 / (1 + np.exp(-raw_predictions))
    predictions = (probabilities > 0.5).astype(int).flatten()

    # Добавим обработку случая, когда в предсказаниях нет одного из классов (для precision/recall)
    try:
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
    except Exception:
        precision, recall, f1 = 0, 0, 0

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

if __name__ == "__main__":
    logging.info("--- [НАЧАЛО] Процесс переобучения и валидации модели ---")

    # 1. Подготовка данных
    all_data = prepare_training_data(config.TRADES_UNIFIED_CSV_PATH)
    if len(all_data) < 100:
        logging.error(f"Недостаточно данных для обучения и валидации ({len(all_data)} сэмплов). Процесс прерван.")
        exit()

    features = np.array([d["features"] for d in all_data])
    targets = np.array([d["target"] for d in all_data])

    # Разделяем на обучающую и тестовую выборки (80/20)
    # Используем stratify для сохранения баланса классов в выборках
    try:
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42, stratify=targets)
    except ValueError: # Если какой-то класс представлен 1 раз, stratify не сработает
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    train_data_split = [{"features": f, "target": t} for f, t in zip(X_train, y_train)]

    # 2. Обучение новой модели ("Претендента")
    logging.info(f"Обучение 'Претендента' на {len(train_data_split)} сэмплах...")
    challenger_model, challenger_scaler = train_golden_model_mlx(train_data_split, num_epochs=50)
    
    # Сохраняем "Претендента" во временные файлы
    challenger_model_path = config.ML_MODEL_PATH.with_suffix(".challenger.safetensors")
    challenger_scaler_path = config.SCALER_PATH.with_suffix(".challenger.pkl")
    save_mlx_checkpoint(challenger_model, challenger_scaler, str(challenger_model_path), str(challenger_scaler_path))

    # 3. Загрузка и оценка моделей
    challenger_inferencer = MLXInferencer(challenger_model_path, challenger_scaler_path)
    challenger_metrics = evaluate_model(challenger_inferencer, X_test, y_test)
    logging.info(f"Метрики 'Претендента' на тестовых данных: {challenger_metrics}")

    champion_metrics = {"accuracy": 0, "f1": 0}
    if config.ML_MODEL_PATH.exists() and config.SCALER_PATH.exists():
        champion_inferencer = MLXInferencer(config.ML_MODEL_PATH, config.SCALER_PATH)
        champion_metrics = evaluate_model(champion_inferencer, X_test, y_test)
        logging.info(f"Метрики 'Чемпиона' на тех же данных: {champion_metrics}")
    else:
        logging.warning("Модель 'Чемпион' не найдена. 'Претендент' будет повышен автоматически.")

    # 4. Сравнение и принятие решения
    # Основной критерий - F1-score, так как он учитывает баланс точности и полноты
    if challenger_metrics["f1"] > champion_metrics["f1"]:
        logging.warning("🏆 'Претендент' показал лучшие результаты! Повышаем до 'Чемпиона'.")
        
        # Архивируем старую модель
        if config.ML_MODEL_PATH.exists():
            archive_path = config.ML_MODEL_PATH.with_suffix(f".archive-{int(time.time())}.safetensors")
            shutil.copy(config.ML_MODEL_PATH, archive_path)
            shutil.copy(config.SCALER_PATH, config.SCALER_PATH.with_suffix(f".archive-{int(time.time())}.pkl"))
            logging.info(f"Старая модель заархивирована в {archive_path.name}")

        # Заменяем старую модель новой
        shutil.move(challenger_model_path, config.ML_MODEL_PATH)
        shutil.move(challenger_scaler_path, config.SCALER_PATH)
        logging.info("Новая модель успешно установлена как рабочая.")
    else:
        logging.info("⚖️ 'Чемпион' остается лучшим. 'Претендент' отклонен.")
        # Удаляем временные файлы претендента
        challenger_model_path.unlink()
        challenger_scaler_path.unlink()

    logging.info("--- [КОНЕЦ] Процесс переобучения и валидации завершен ---")