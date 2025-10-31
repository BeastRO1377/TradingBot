# Trading Bot Dockerfile
# Многоэтапная сборка для оптимизации размера образа

# Этап 1: Базовый образ с Python
FROM python:3.11-slim as base

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создаем пользователя для безопасности
RUN groupadd -r botuser && useradd -r -g botuser botuser

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

# Создаем виртуальное окружение и устанавливаем зависимости
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Этап 2: Финальный образ
FROM python:3.11-slim

# Устанавливаем только необходимые системные пакеты
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создаем пользователя
RUN groupadd -r botuser && useradd -r -g botuser botuser

# Копируем виртуальное окружение из предыдущего этапа
COPY --from=base /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем исходный код
COPY --chown=botuser:botuser . .

# Создаем необходимые директории
RUN mkdir -p /app/logs /app/data /app/models && \
    chown -R botuser:botuser /app

# Переключаемся на пользователя botuser
USER botuser

# Переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV BOT_MODE=trading

# Открываем порт (если нужен веб-интерфейс)
EXPOSE 8080

# Создаем volume для данных
VOLUME ["/app/data", "/app/logs"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)" || exit 1

# Команда по умолчанию
CMD ["python", "launcher.py", "--mode", "trading"]

# Метаданные
LABEL maintainer="Trading Bot Team"
LABEL description="Trading Bot with ML capabilities"
LABEL version="1.0.0"
























