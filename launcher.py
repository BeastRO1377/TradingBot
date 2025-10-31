#!/usr/bin/env python3
"""
Trading Bot Launcher
Универсальный запускатор торгового бота с поддержкой различных режимов работы
"""

import asyncio
import argparse
import logging
import os
import sys
import signal
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess
import time

# Добавляем текущую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('launcher.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class BotLauncher:
    """Класс для управления запуском торгового бота"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.processes: List[subprocess.Popen] = []
        self.running = False
        
    def check_dependencies(self) -> bool:
        """Проверяет наличие всех необходимых зависимостей"""
        logger.info("Проверка зависимостей...")
        
        # Проверяем Python версию
        if sys.version_info < (3, 8):
            logger.error("Требуется Python 3.8 или выше")
            return False
            
        # Проверяем наличие основных файлов
        required_files = [
            'main.py',
            'bot_core.py', 
            'config.py',
            'requirements.txt',
            'user_state.json'
        ]
        
        for file in required_files:
            if not (self.root_dir / file).exists():
                logger.error(f"Отсутствует обязательный файл: {file}")
                return False
                
        # Проверяем наличие виртуального окружения или устанавливаем зависимости
        venv_path = self.root_dir / '.venv'
        if not venv_path.exists():
            logger.info("Создание виртуального окружения...")
            try:
                subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True)
                logger.info("Виртуальное окружение создано")
            except subprocess.CalledProcessError as e:
                logger.error(f"Ошибка создания виртуального окружения: {e}")
                return False
                
        # Устанавливаем зависимости
        pip_path = venv_path / 'bin' / 'pip' if os.name != 'nt' else venv_path / 'Scripts' / 'pip.exe'
        if pip_path.exists():
            logger.info("Установка зависимостей...")
            try:
                subprocess.run([str(pip_path), 'install', '-r', 'requirements.txt'], 
                             cwd=self.root_dir, check=True)
                logger.info("Зависимости установлены")
            except subprocess.CalledProcessError as e:
                logger.error(f"Ошибка установки зависимостей: {e}")
                return False
                
        return True
        
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Загружает конфигурацию бота"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
        
    def start_bot(self, mode: str = 'trading', config: Optional[Dict[str, Any]] = None) -> bool:
        """Запускает бота в указанном режиме"""
        logger.info(f"Запуск бота в режиме: {mode}")
        
        # Определяем команду запуска
        python_path = self.root_dir / '.venv' / 'bin' / 'python'
        if os.name == 'nt':  # Windows
            python_path = self.root_dir / '.venv' / 'Scripts' / 'python.exe'
            
        if not python_path.exists():
            python_path = sys.executable
            
        cmd = [str(python_path), 'main.py']
        
        # Добавляем аргументы в зависимости от режима
        if mode == 'training':
            cmd.append('--train')
        elif mode == 'demo':
            cmd.extend(['--mode', 'demo'])
        elif mode == 'real':
            cmd.extend(['--mode', 'real'])
            
        # Устанавливаем переменные окружения
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.root_dir)
        env['PYTHONUNBUFFERED'] = '1'
        
        # Применяем конфигурацию
        if config:
            for key, value in config.items():
                env[f'BOT_{key.upper()}'] = str(value)
                
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.root_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes.append(process)
            self.running = True
            
            # Запускаем мониторинг вывода
            asyncio.create_task(self._monitor_process(process))
            
            logger.info(f"Бот запущен с PID: {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска бота: {e}")
            return False
            
    async def _monitor_process(self, process: subprocess.Popen):
        """Мониторит вывод процесса бота"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.rstrip())
        except Exception as e:
            logger.error(f"Ошибка мониторинга процесса: {e}")
            
    def stop_bot(self):
        """Останавливает все запущенные процессы бота"""
        logger.info("Остановка бота...")
        self.running = False
        
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Принудительное завершение процесса...")
                process.kill()
            except Exception as e:
                logger.error(f"Ошибка остановки процесса: {e}")
                
        self.processes.clear()
        logger.info("Бот остановлен")
        
    def setup_signal_handlers(self):
        """Настраивает обработчики сигналов для корректного завершения"""
        def signal_handler(signum, frame):
            logger.info(f"Получен сигнал {signum}, завершение работы...")
            self.stop_bot()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def run_interactive(self):
        """Интерактивный режим запуска"""
        print("\n🤖 Trading Bot Launcher")
        print("=" * 50)
        
        while True:
            print("\nДоступные режимы:")
            print("1. Торговля (реальный режим)")
            print("2. Торговля (демо режим)")
            print("3. Обучение модели")
            print("4. Проверка зависимостей")
            print("5. Выход")
            
            choice = input("\nВыберите режим (1-5): ").strip()
            
            if choice == '1':
                if self.check_dependencies():
                    self.setup_signal_handlers()
                    self.start_bot('real')
                    try:
                        while self.running:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        self.stop_bot()
                        
            elif choice == '2':
                if self.check_dependencies():
                    self.setup_signal_handlers()
                    self.start_bot('demo')
                    try:
                        while self.running:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        self.stop_bot()
                        
            elif choice == '3':
                if self.check_dependencies():
                    self.setup_signal_handlers()
                    self.start_bot('training')
                    try:
                        while self.running:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        self.stop_bot()
                        
            elif choice == '4':
                self.check_dependencies()
                
            elif choice == '5':
                print("До свидания!")
                break
                
            else:
                print("Неверный выбор, попробуйте снова")

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Trading Bot Launcher')
    parser.add_argument('--mode', choices=['trading', 'demo', 'training'], 
                       default='trading', help='Режим работы бота')
    parser.add_argument('--config', type=str, help='Путь к файлу конфигурации')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Интерактивный режим')
    parser.add_argument('--check-deps', action='store_true', 
                       help='Только проверить зависимости')
    
    args = parser.parse_args()
    
    launcher = BotLauncher()
    
    if args.check_deps:
        if launcher.check_dependencies():
            print("✅ Все зависимости установлены")
            sys.exit(0)
        else:
            print("❌ Проблемы с зависимостями")
            sys.exit(1)
            
    if args.interactive:
        launcher.run_interactive()
    else:
        if launcher.check_dependencies():
            config = launcher.load_config(args.config)
            launcher.setup_signal_handlers()
            launcher.start_bot(args.mode, config)
            
            try:
                while launcher.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                launcher.stop_bot()

if __name__ == '__main__':
    main()
























