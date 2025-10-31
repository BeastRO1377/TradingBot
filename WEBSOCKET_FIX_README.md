# WebSocket Connection Fix

## Проблема
1. **Публичный WebSocket**: Постоянные разрывы с ошибками `ping/pong timed out` каждые 1-2 минуты
2. **Приватный WebSocket**: Нестабильные соединения в демо-режиме с частыми разрывами

## Причины
1. **Слишком короткий ping_timeout** (15 секунд)
2. **Неправильное соотношение ping_interval/ping_timeout** (должно быть ping_interval > ping_timeout)
3. **Отсутствие exponential backoff** при переподключении
4. **Недостаточный мониторинг** состояния соединения
5. **Плохая обработка ошибок** в callback функциях
6. **Отсутствие логики переподключения** для приватного WebSocket

## Исправления

### 1. Улучшенные настройки WebSocket
```python
# Было
ping_interval=30, ping_timeout=15

# Стало
ping_interval=60, ping_timeout=30  # ping_interval > ping_timeout (требование библиотеки)
```

### 2. Exponential Backoff для переподключения
```python
def _calculate_reconnect_delay(self) -> float:
    """Вычисляет задержку переподключения с exponential backoff"""
    if self.reconnect_attempts == 0:
        return self.base_reconnect_delay
    
    delay = min(
        self.base_reconnect_delay * (2 ** min(self.reconnect_attempts - 1, 8)),
        self.max_reconnect_delay
    )
    # Добавляем случайную jitter для избежания thundering herd
    jitter = random.uniform(0.5, 1.5)
    return delay * jitter
```

### 3. Мониторинг состояния соединения
```python
def _is_connection_healthy(self) -> bool:
    """Проверяет здоровье соединения"""
    if not self.ws or not self.ws.is_connected():
        return False
    
    # Проверяем, что соединение не слишком старое (максимум 24 часа)
    if self.connection_start_time and (time.time() - self.connection_start_time) > 86400:
        logger.info("Соединение слишком старое, переподключаемся...")
        return False
    
    return True
```

### 4. Улучшенная обработка ошибок
```python
def _on_message(msg):
    try:
        # Записываем получение сообщения в монитор
        self.monitor.record_message(self.connection_name)
        
        if not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.route_message(msg), self.loop)
    except asyncio.CancelledError:
        # Игнорируем отмененные задачи
        pass
    except Exception as e:
        logger.warning(f"PublicWS callback error: {e}")
        self.monitor.record_error(self.connection_name, str(e))
        # Не прерываем соединение из-за ошибок в callback
```

### 5. Исправление приватного WebSocket
```python
async def setup_private_ws(self):
    """Настраивает приватный WebSocket с улучшенной логикой переподключения"""
    reconnect_attempts = 0
    max_reconnect_attempts = 10
    base_reconnect_delay = 5
    max_reconnect_delay = 300  # 5 минут
    
    while True:
        try:
            # Создаем соединение с правильными настройками
            self.ws_private = WebSocket(
                testnet=False, demo=self.mode == "demo", channel_type="private",
                api_key=self.api_key, api_secret=self.api_secret,
                ping_interval=60, ping_timeout=30, restart_on_error=True, retries=200
            )
            
            # Мониторим соединение и переподключаемся при разрыве
            while self.ws_private.is_connected():
                await asyncio.sleep(1)
                
        except Exception as e:
            # Exponential backoff для переподключения
            delay = min(base_reconnect_delay * (2 ** min(reconnect_attempts - 1, 8)), max_reconnect_delay)
            await asyncio.sleep(delay)
```

### 6. Система мониторинга WebSocket
- **websocket_monitor.py** - Полноценная система мониторинга
- Отслеживание ping/pong, сообщений, ошибок, переподключений
- Диагностика проблем с соединениями
- Автоматические отчеты о состоянии

## Новые возможности

### 1. Автоматическое управление переподключением
- Максимум 10 попыток переподключения
- Exponential backoff с jitter
- Пауза 5 минут после исчерпания попыток

### 2. Мониторинг в реальном времени
- Отслеживание времени работы соединения
- Подсчет сообщений и ошибок
- Автоматическая диагностика проблем

### 3. Улучшенное логирование
- Статус соединения каждые 5 минут
- Детальная информация о переподключениях
- Отчеты о здоровье соединения

## Тестирование

### Запуск тестов
```bash
# Тест стабильности соединения
python test_websocket_fix.py

# Или через launcher
python launcher.py --mode trading
```

### Мониторинг
```python
from websocket_monitor import get_monitor, print_connection_report

# Получить монитор
monitor = get_monitor()

# Вывести отчет
print_connection_report()

# Получить статистику
stats = monitor.get_summary()
```

## Ожидаемые результаты

### До исправления
- Разрывы соединения каждые 1-2 минуты
- Ошибки `ping/pong timed out`
- Частые переподключения
- Потеря данных

### После исправления
- Стабильные соединения на часы
- Редкие переподключения только при реальных проблемах
- Автоматическое восстановление
- Полный мониторинг состояния

## Конфигурация

### Настройки в data_manager.py
```python
# Настройки переподключения
self.max_reconnect_attempts = 10
self.base_reconnect_delay = 5
self.max_reconnect_delay = 300  # 5 минут

# Настройки ping/pong
self.ping_interval = 60  # Ping каждые 60 секунд
self.ping_timeout = 30   # Timeout 30 секунд (должен быть меньше ping_interval)
```

### Настройки в bot_core.py
```python
# Приватный WebSocket
ping_interval=60, ping_timeout=30  # ping_interval > ping_timeout
```

## Диагностика проблем

### Проверка состояния
```python
from websocket_monitor import diagnose_connection_issues

issues = diagnose_connection_issues()
print(issues)
```

### Анализ логов
```bash
# Поиск ошибок ping/pong
grep "ping/pong timed out" bot.log

# Поиск переподключений
grep "Переподключение" bot.log

# Анализ частоты ошибок
grep "ERROR" bot.log | wc -l
```

## Мониторинг в продакшене

### 1. Автоматические отчеты
```python
# В main.py добавить
from websocket_monitor import get_monitor

async def periodic_health_check():
    monitor = get_monitor()
    while True:
        await asyncio.sleep(300)  # Каждые 5 минут
        summary = monitor.get_summary()
        if summary['healthy_connections'] < summary['total_connections']:
            logger.warning(f"Проблемы с WebSocket: {summary}")
```

### 2. Алерты
```python
# Настройка алертов при проблемах
def check_websocket_health():
    monitor = get_monitor()
    issues = diagnose_connection_issues()
    
    if issues['high_error_rate']:
        send_alert("Высокий уровень ошибок WebSocket")
    
    if issues['frequent_reconnects']:
        send_alert("Частые переподключения WebSocket")
```

## Заключение

Исправления должны значительно улучшить стабильность WebSocket соединений:

1. **Увеличенный timeout** предотвращает ложные разрывы
2. **Exponential backoff** снижает нагрузку на сервер
3. **Мониторинг** позволяет быстро выявлять проблемы
4. **Улучшенная обработка ошибок** повышает надежность

Ожидается снижение количества разрывов соединения на 90%+ и улучшение общей стабильности системы.

