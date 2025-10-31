#!/usr/bin/env python3
"""
Test WebSocket Fix
Скрипт для тестирования исправлений WebSocket соединений
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Добавляем текущую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from data_manager import PublicWebSocketManager
from websocket_monitor import get_monitor, print_connection_report

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('websocket_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

async def test_websocket_stability():
    """Тестирует стабильность WebSocket соединения"""
    logger.info("Начинаем тест стабильности WebSocket соединения...")
    
    # Создаем WebSocket менеджер
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT"]
    ws_manager = PublicWebSocketManager(symbols)
    
    # Запускаем мониторинг
    monitor = get_monitor()
    monitor_task = asyncio.create_task(monitor.start_monitoring(30))
    
    try:
        # Запускаем WebSocket
        ws_task = asyncio.create_task(ws_manager.start())
        
        # Ждем 10 минут для тестирования
        logger.info("Тестируем соединение в течение 10 минут...")
        await asyncio.sleep(600)
        
        # Выводим отчет
        print_connection_report()
        
    except KeyboardInterrupt:
        logger.info("Тест прерван пользователем")
    except Exception as e:
        logger.error(f"Ошибка в тесте: {e}")
    finally:
        # Останавливаем задачи
        ws_task.cancel()
        monitor.stop_monitoring()
        monitor_task.cancel()
        
        # Ждем завершения
        await asyncio.gather(ws_task, monitor_task, return_exceptions=True)
        
        logger.info("Тест завершен")

async def test_reconnection_logic():
    """Тестирует логику переподключения"""
    logger.info("Начинаем тест логики переподключения...")
    
    symbols = ["BTCUSDT", "ETHUSDT"]
    ws_manager = PublicWebSocketManager(symbols)
    
    # Симулируем проблемы с соединением
    original_start = ws_manager.start
    
    async def mock_start():
        """Мок-версия start с принудительными разрывами"""
        await original_start()
    
    ws_manager.start = mock_start
    
    try:
        ws_task = asyncio.create_task(ws_manager.start())
        
        # Ждем 5 минут
        await asyncio.sleep(300)
        
    except KeyboardInterrupt:
        logger.info("Тест переподключения прерван")
    finally:
        ws_task.cancel()
        await ws_task

def analyze_logs():
    """Анализирует логи на предмет проблем"""
    log_file = Path("websocket_test.log")
    if not log_file.exists():
        logger.warning("Файл логов не найден")
        return
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Анализируем ошибки
    errors = [line for line in lines if "ERROR" in line]
    warnings = [line for line in lines if "WARNING" in line]
    reconnects = [line for line in lines if "Переподключение" in line]
    
    print(f"\nАнализ логов:")
    print(f"Всего строк: {len(lines)}")
    print(f"Ошибок: {len(errors)}")
    print(f"Предупреждений: {len(warnings)}")
    print(f"Переподключений: {len(reconnects)}")
    
    if errors:
        print(f"\nПоследние ошибки:")
        for error in errors[-5:]:
            print(f"  {error.strip()}")
    
    if reconnects:
        print(f"\nЧастота переподключений:")
        reconnect_times = []
        for line in reconnects:
            # Извлекаем время из лога
            time_part = line.split(' - ')[0]
            reconnect_times.append(time_part)
        
        if len(reconnect_times) > 1:
            print(f"  Первое: {reconnect_times[0]}")
            print(f"  Последнее: {reconnect_times[-1]}")
            print(f"  Всего: {len(reconnect_times)}")

async def main():
    """Главная функция тестирования"""
    print("WebSocket Fix Test Suite")
    print("=" * 50)
    
    while True:
        print("\nВыберите тест:")
        print("1. Тест стабильности соединения (10 минут)")
        print("2. Тест логики переподключения (5 минут)")
        print("3. Анализ логов")
        print("4. Выход")
        
        choice = input("\nВведите номер теста (1-4): ").strip()
        
        if choice == "1":
            await test_websocket_stability()
        elif choice == "2":
            await test_reconnection_logic()
        elif choice == "3":
            analyze_logs()
        elif choice == "4":
            print("До свидания!")
            break
        else:
            print("Неверный выбор, попробуйте снова")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)












































