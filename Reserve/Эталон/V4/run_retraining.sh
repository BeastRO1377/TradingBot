#!/bin/bash

# --- НАСТРОЙТЕ ЭТИ ПУТИ ---
PROJECT_DIR="/Users/anatolytamilin/Downloads/TradingBot"
# --- КОНЕЦ НАСТРОЕК ---

# Переходим в директорию проекта
cd "$PROJECT_DIR"

# Активируем виртуальное окружение (если вы его используете, что крайне рекомендуется)
# Если ваше окружение называется не "venv", измените путь.
source "$PROJECT_DIR/.venv/bin/activate"

# Запускаем скрипт переобучения с помощью python из виртуального окружения
# и перенаправляем весь вывод (стандартный и ошибки) в лог-файл
echo "--- Запуск переобучения: $(date) ---" >> retraining.log
python3 "$PROJECT_DIR/retrain_and_validate.py" >> retraining.log 2>&1
echo "--- Переобучение завершено: $(date) ---" >> retraining.log

# Деактивируем виртуальное окружение
deactivate