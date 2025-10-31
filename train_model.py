import logging
from pathlib import Path

import numpy as np

# --- monkey patch for numpy NaN issue with pandas-ta (legacy) ---
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import config
from ai_ml import (
    build_training_samples_from_csv,
    save_mlx_checkpoint,
    train_golden_model_mlx,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    csv_path = Path(config.TRADES_UNIFIED_CSV_PATH)
    samples = build_training_samples_from_csv(csv_path)

    if len(samples) < getattr(config, "ML_AUTO_TRAIN_MIN_SAMPLES", 200):
        logging.error(
            "Недостаточно данных для обучения (%d < %d).",
            len(samples),
            getattr(config, "ML_AUTO_TRAIN_MIN_SAMPLES", 200),
        )
        return

    epochs = getattr(config, "ML_AUTO_TRAIN_EPOCHS", 45)
    logging.info(
        "Старт обучения MLX-модели: samples=%d, epochs=%d.",
        len(samples),
        epochs,
    )

    model, scaler = train_golden_model_mlx(samples, num_epochs=epochs)
    save_mlx_checkpoint(model, scaler)

    logging.info("Обучение завершено. Модель и скейлер сохранены.")


if __name__ == "__main__":
    main()
