# utils.py
import json
import uuid
import os
import tempfile
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
import logging
from datetime import datetime
import re
from typing import Any
from functools import wraps
import asyncio

# Импортируем константы из конфига
import config

logger = logging.getLogger(__name__)

def safe_to_float(val, default=0.0) -> float:
    """Безопасно конвертирует значение в float."""
    try:
        return float(val)
    except (ValueError, TypeError, AttributeError):
        return default

def async_retry(max_retries=3, delay=2, backoff=2):
    """
    Декоратор для асинхронных функций, который повторяет вызов
    в случае исключений.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Функция {func.__name__} не удалась после {max_retries} попыток. Ошибка: {e}", exc_info=True)
                        raise
                    logger.warning(f"Попытка {attempt + 1}/{max_retries} для {func.__name__} не удалась. Ошибка: {e}. Повтор через {current_delay} сек...")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


def new_cid() -> str:
    """Генерирует короткий уникальный ID для логов и ордеров."""
    return uuid.uuid4().hex[:8]

def j(obj, maxlen=600) -> str:
    """Безопасно конвертирует объект в JSON-строку для логов, усекая длинные структуры."""
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        return (s[:maxlen] + "…") if len(s) > maxlen else s
    except Exception:
        return str(obj)[:maxlen]

def safe_parse_json(text: str | None, default: Any = None) -> Any:
    """
    Пытается распарсить JSON из ответа LLM/Ollama.
    - Аккуратно обрабатывает пустую строку
    - Срезает кодовые блоки ```json ... ```
    - Если модель вернула пояснительный текст вокруг JSON, пытается вытащить первый блок {...}
    - Возвращает default при любой ошибке
    """
    if text is None:
        return default

    s = text.strip()

    # Снимаем ограждение ```json ... ```
    if s.startswith("```"):
        # убираем первую строку с ```/```json
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
        # убираем завершающие ```
        s = re.sub(r"\s*```$", "", s).strip()

    # Прямая попытка
    try:
        return json.loads(s)
    except Exception:
        pass

    # Если вокруг JSON есть текст – вытащим первый блок {...}
    try:
        start = s.find("{")
        end   = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = s[start:end+1]
            return json.loads(candidate)
    except Exception:
        pass

    # Ничего не вышло
    return default

def _atomic_json_write(path: Path, data: any) -> None:
    """Атомарно записывает JSON в файл через временный файл."""
    try:
        dirname = path.parent
        dirname.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dirname, prefix=".tmp_", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        logger.error(f"Atomic write to {path} failed: {e}")

def _append_trades_unified(row: dict) -> None:
    """
    [ИСПРАВЛЕННАЯ ВЕРСИЯ] Добавляет строку в CSV-файл с историей всех сделок,
    используя жестко заданный список заголовков из config.py.
    """
    try:
        # Получаем эталонный список заголовков из конфига
        headers = config.TRADES_UNIFIED_CSV_HEADERS
        path = config.TRADES_UNIFIED_CSV_PATH
        file_exists = path.is_file()

        if file_exists:
            # Проверяем, что хедер в файле соответствует актуальному списку полей.
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                try:
                    existing_header = next(reader)
                except StopIteration:
                    existing_header = []

            if list(existing_header) != headers:
                # Переписываем файл с новым хедером, сохраняя старые значения.
                with path.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    old_rows = [dict(r) for r in reader]

                with path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
                    writer.writeheader()
                    for old_row in old_rows:
                        upgraded = {key: old_row.get(key, "") for key in headers}
                        writer.writerow(upgraded)

                file_exists = True

        with path.open("a", newline="", encoding="utf-8") as f:
            # Используем predefined headers. extrasaction='ignore' проигнорирует
            # любые лишние ключи в `row`, не ломая структуру.
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)
            
    except Exception as e:
        logger.error(f"Failed to append to trades_unified.csv: {e}")


def compute_pct(candles_deque, minutes: int) -> float:
    """Рассчитывает процентное изменение цены за N минут."""
    data = list(candles_deque)
    if len(data) < minutes + 1:
        return 0.0
    old_close = safe_to_float(data[-minutes - 1].get("closePrice", 0))
    new_close = safe_to_float(data[-1].get("closePrice", 0))
    if old_close <= 0:
        return 0.0
    return (new_close - old_close) / old_close * 100.0

def sum_last_vol(candles_deque, minutes: int) -> float:
    """Суммирует объем за последние N свечей."""
    data = list(candles_deque)[-minutes:]
    return sum(safe_to_float(c.get("volume", 0)) for c in data)

def calc_pnl(entry_side: str, entry_price: any, exit_price: any, qty: any) -> float:
    """Безопасно рассчитывает PnL в USDT, используя Decimal для точности."""
    try:
        d_entry_price = Decimal(str(entry_price))
        d_exit_price = Decimal(str(exit_price))
        d_qty = Decimal(str(qty))

        if entry_side.lower() == "buy":
            pnl = (d_exit_price - d_entry_price) * d_qty
        else:
            pnl = (d_entry_price - d_exit_price) * d_qty
        return float(pnl)
    except (InvalidOperation, TypeError, ValueError) as e:
        logger.error(f"PNL calculation error: entry={entry_price}, exit={exit_price}, qty={qty}. Error: {e}")
        return 0.0

def log_for_finetune(prompt: str, pnl_pct: float, source: str):
    """Записывает данные, необходимые для дообучения модели."""
    log_file = config.FINETUNE_LOG_FILE
    is_new = not log_file.exists()
    try:
        with log_file.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            if is_new:
                writer.writerow(["timestamp", "source", "pnl_pct", "prompt"])
            writer.writerow([datetime.utcnow().isoformat(), source, pnl_pct, prompt])
    except Exception as e:
        logger.error(f"[FineTuneLog] Ошибка записи лога для дообучения: {e}")

import numpy as np

class SimpleClusterAnalyzer:
    def __init__(self, depth=60):
        self.depth = depth

    def analyze(self, orderbook: dict) -> dict:
        bids = orderbook.get("bids", {})
        asks = orderbook.get("asks", {})

        bid_prices = np.array([float(p) for p in bids.keys()])
        bid_sizes  = np.array([float(s) for s in bids.values()])

        ask_prices = np.array([float(p) for p in asks.keys()])
        ask_sizes  = np.array([float(s) for s in asks.values()])

        top_bid_cluster_idx = bid_sizes.argmax() if bid_sizes.size > 0 else None
        top_ask_cluster_idx = ask_sizes.argmax() if ask_sizes.size > 0 else None

        return {
            "top_bid_price": float(bid_prices[top_bid_cluster_idx]) if top_bid_cluster_idx is not None else None,
            "top_bid_size":  float(bid_sizes[top_bid_cluster_idx]) if top_bid_cluster_idx is not None else None,
            "top_ask_price": float(ask_prices[top_ask_cluster_idx]) if top_ask_cluster_idx is not None else None,
            "top_ask_size":  float(ask_sizes[top_ask_cluster_idx]) if top_ask_cluster_idx is not None else None,
        }
