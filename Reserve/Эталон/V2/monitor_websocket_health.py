#!/usr/bin/env python3
"""
WebSocket Health Monitor
Скрипт для мониторинга здоровья WebSocket соединений в реальном времени
"""

import asyncio
import logging
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Добавляем текущую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from websocket_monitor import get_monitor, print_connection_report, diagnose_connection_issues

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebSocketHealthMonitor:
    """Монитор здоровья WebSocket соединений"""
    
    def __init__(self):
        self.monitor = get_monitor()
        self.running = False
        self.stats_history = []
        self.max_history = 100
        
    def log_health_status(self):
        """Логирует текущий статус здоровья соединений"""
        summary = self.monitor.get_summary()
        issues = diagnose_connection_issues()
        
        # Сохраняем статистику
        self.stats_history.append({
            'timestamp': time.time(),
            'healthy_connections': summary['healthy_connections'],
            'total_connections': summary['total_connections'],
            'total_errors': summary['total_errors'],
            'total_reconnects': summary['total_reconnects']
        })
        
        # Ограничиваем размер истории
        if len(self.stats_history) > self.max_history:
            self.stats_history.pop(0)
        
        # Логируем статус
        healthy_ratio = summary['healthy_connections'] / max(summary['total_connections'], 1)
        
        if healthy_ratio >= 0.8:
            status = "🟢 ОТЛИЧНО"
        elif healthy_ratio >= 0.6:
            status = "🟡 УДОВЛЕТВОРИТЕЛЬНО"
        else:
            status = "🔴 ПРОБЛЕМЫ"
        
        logger.info(f"WebSocket Health: {status} ({summary['healthy_connections']}/{summary['total_connections']} соединений)")
        
        # Логируем проблемы
        if issues['high_error_rate']:
            logger.warning(f"Высокий уровень ошибок: {len(issues['high_error_rate'])} соединений")
        
        if issues['frequent_reconnects']:
            logger.warning(f"Частые переподключения: {len(issues['frequent_reconnects'])} соединений")
        
        if issues['no_recent_activity']:
            logger.warning(f"Нет активности: {len(issues['no_recent_activity'])} соединений")
        
        if issues['ping_pong_issues']:
            logger.warning(f"Проблемы ping/pong: {len(issues['ping_pong_issues'])} соединений")
    
    def get_health_trend(self, minutes: int = 10) -> dict:
        """Анализирует тренд здоровья за последние N минут"""
        cutoff_time = time.time() - (minutes * 60)
        recent_stats = [s for s in self.stats_history if s['timestamp'] > cutoff_time]
        
        if len(recent_stats) < 2:
            return {'trend': 'insufficient_data', 'message': 'Недостаточно данных для анализа'}
        
        # Анализируем тренд
        first = recent_stats[0]
        last = recent_stats[-1]
        
        healthy_change = last['healthy_connections'] - first['healthy_connections']
        error_change = last['total_errors'] - first['total_errors']
        reconnect_change = last['total_reconnects'] - first['total_reconnects']
        
        if healthy_change > 0 and error_change < 5 and reconnect_change < 3:
            trend = 'improving'
            message = 'Состояние улучшается'
        elif healthy_change < 0 or error_change > 10 or reconnect_change > 5:
            trend = 'degrading'
            message = 'Состояние ухудшается'
        else:
            trend = 'stable'
            message = 'Состояние стабильно'
        
        return {
            'trend': trend,
            'message': message,
            'healthy_change': healthy_change,
            'error_change': error_change,
            'reconnect_change': reconnect_change,
            'data_points': len(recent_stats)
        }
    
    def print_detailed_report(self):
        """Выводит детальный отчет о состоянии соединений"""
        print("\n" + "="*80)
        print("DETAILED WEBSOCKET HEALTH REPORT")
        print("="*80)
        
        summary = self.monitor.get_summary()
        issues = diagnose_connection_issues()
        
        print(f"Время отчета: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Всего соединений: {summary['total_connections']}")
        print(f"Здоровых соединений: {summary['healthy_connections']}")
        print(f"Всего сообщений: {summary['total_messages']}")
        print(f"Всего переподключений: {summary['total_reconnects']}")
        print(f"Всего ошибок: {summary['total_errors']}")
        
        # Анализ тренда
        trend = self.get_health_trend(10)
        print(f"\nТренд за последние 10 минут: {trend['message']}")
        if trend['data_points'] > 0:
            print(f"Изменение здоровых соединений: {trend['healthy_change']:+d}")
            print(f"Изменение ошибок: {trend['error_change']:+d}")
            print(f"Изменение переподключений: {trend['reconnect_change']:+d}")
        
        # Детали по соединениям
        print(f"\nДетали по соединениям:")
        for name, conn_info in summary['connections'].items():
            status = "✅ ЗДОРОВО" if conn_info['healthy'] else "❌ ПРОБЛЕМЫ"
            uptime_hours = conn_info['uptime'] / 3600
            print(f"  {name}:")
            print(f"    Статус: {status}")
            print(f"    Время работы: {uptime_hours:.1f}ч")
            print(f"    Сообщений: {conn_info['messages_received']}")
            print(f"    Переподключений: {conn_info['reconnects']}")
            print(f"    Ошибок: {conn_info['errors']}")
            if conn_info['last_error']:
                print(f"    Последняя ошибка: {conn_info['last_error']}")
        
        # Проблемы
        if any(issues.values()):
            print(f"\nОбнаруженные проблемы:")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    print(f"  {issue_type}: {len(issue_list)} соединений")
                    for issue in issue_list[:3]:  # Показываем только первые 3
                        print(f"    - {issue}")
                    if len(issue_list) > 3:
                        print(f"    ... и еще {len(issue_list) - 3}")
        
        print("="*80)
    
    async def start_monitoring(self, interval: int = 30):
        """Запускает мониторинг с заданным интервалом"""
        logger.info(f"Запуск мониторинга WebSocket соединений (интервал: {interval}с)")
        self.running = True
        
        try:
            while self.running:
                self.log_health_status()
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Мониторинг прерван пользователем")
        except Exception as e:
            logger.error(f"Ошибка в мониторинге: {e}")
        finally:
            self.running = False
            logger.info("Мониторинг остановлен")
    
    def stop_monitoring(self):
        """Останавливает мониторинг"""
        self.running = False

async def main():
    """Главная функция"""
    monitor = WebSocketHealthMonitor()
    
    print("WebSocket Health Monitor")
    print("=" * 50)
    
    while True:
        print("\nВыберите действие:")
        print("1. Запустить мониторинг (30с интервал)")
        print("2. Запустить мониторинг (60с интервал)")
        print("3. Показать текущий статус")
        print("4. Показать детальный отчет")
        print("5. Анализ тренда за 10 минут")
        print("6. Выход")
        
        choice = input("\nВведите номер (1-6): ").strip()
        
        if choice == "1":
            await monitor.start_monitoring(30)
        elif choice == "2":
            await monitor.start_monitoring(60)
        elif choice == "3":
            monitor.log_health_status()
        elif choice == "4":
            monitor.print_detailed_report()
        elif choice == "5":
            trend = monitor.get_health_trend(10)
            print(f"\nТренд за последние 10 минут:")
            print(f"  {trend['message']}")
            if trend['data_points'] > 0:
                print(f"  Здоровые соединения: {trend['healthy_change']:+d}")
                print(f"  Ошибки: {trend['error_change']:+d}")
                print(f"  Переподключения: {trend['reconnect_change']:+d}")
                print(f"  Точек данных: {trend['data_points']}")
        elif choice == "6":
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































