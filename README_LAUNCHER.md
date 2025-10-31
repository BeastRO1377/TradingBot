# Trading Bot Launcher

Универсальная система запуска торгового бота с поддержкой различных режимов работы и развертывания.

## 🚀 Быстрый старт

### 1. Простой запуск
```bash
# Интерактивный режим
python launcher.py --interactive

# Или через shell-скрипт
./start_bot.sh
```

### 2. Режимы работы
```bash
# Торговля (реальный режим)
python launcher.py --mode trading

# Демо режим
python launcher.py --mode demo

# Обучение модели
python launcher.py --mode training

# Проверка зависимостей
python launcher.py --check-deps
```

## 📋 Компоненты системы

### 1. `launcher.py` - Основной запускатор
- ✅ Автоматическая проверка зависимостей
- ✅ Создание виртуального окружения
- ✅ Установка пакетов
- ✅ Мониторинг процессов
- ✅ Graceful shutdown
- ✅ Интерактивный режим

### 2. `start_bot.sh` - Shell-скрипт
- ✅ Цветной вывод
- ✅ Проверка зависимостей
- ✅ Фоновый режим (daemon)
- ✅ Управление процессами
- ✅ Логирование
- ✅ Статус и перезапуск

### 3. `Dockerfile` - Контейнеризация
- ✅ Многоэтапная сборка
- ✅ Оптимизация размера
- ✅ Безопасность (non-root user)
- ✅ Health checks
- ✅ Volume mounts

### 4. `docker-compose.yml` - Оркестрация
- ✅ Торговый бот
- ✅ Redis для кэширования
- ✅ Nginx для проксирования
- ✅ Prometheus для мониторинга
- ✅ Grafana для визуализации

### 5. `trading-bot.service` - Systemd сервис
- ✅ Автозапуск
- ✅ Автоматический перезапуск
- ✅ Ограничения ресурсов
- ✅ Безопасность
- ✅ Логирование

## 🛠 Установка и настройка

### Локальная установка

1. **Клонирование и настройка:**
```bash
git clone <repository-url>
cd TradingBot
chmod +x start_bot.sh install_service.sh
```

2. **Проверка зависимостей:**
```bash
./start_bot.sh --check-deps
```

3. **Установка зависимостей:**
```bash
./start_bot.sh --install-deps
```

4. **Запуск:**
```bash
./start_bot.sh trading
```

### Docker установка

1. **Сборка образа:**
```bash
docker build -t trading-bot .
```

2. **Запуск контейнера:**
```bash
docker run -d \
  --name trading-bot \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/user_state.json:/app/user_state.json:ro \
  trading-bot
```

3. **Или через docker-compose:**
```bash
docker-compose up -d
```

### Systemd сервис

1. **Установка сервиса:**
```bash
sudo ./install_service.sh
```

2. **Управление:**
```bash
# Запуск
sudo systemctl start trading-bot

# Остановка
sudo systemctl stop trading-bot

# Статус
sudo systemctl status trading-bot

# Логи
sudo journalctl -u trading-bot -f
```

## 📊 Мониторинг

### Логи
```bash
# Основные логи
tail -f bot.log

# Systemd логи
sudo journalctl -u trading-bot -f

# Docker логи
docker logs -f trading-bot
```

### Статус
```bash
# Проверка статуса
./start_bot.sh --status

# Или через systemctl
sudo systemctl status trading-bot
```

### Веб-интерфейс (если настроен)
- Основной интерфейс: http://localhost:8080
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

## 🔧 Конфигурация

### Переменные окружения
```bash
export LOG_LEVEL=INFO
export BOT_MODE=trading
export BOT_PORT=8080
export PYTHONPATH=/path/to/bot
```

### Файлы конфигурации
- `config.py` - Основная конфигурация
- `user_state.json` - Пользователи и настройки
- `.env` - Переменные окружения

## 🚨 Устранение неполадок

### Проблемы с зависимостями
```bash
# Переустановка зависимостей
./start_bot.sh --install-deps

# Очистка виртуального окружения
rm -rf .venv
./start_bot.sh --install-deps
```

### Проблемы с правами доступа
```bash
# Исправление прав
chmod +x launcher.py start_bot.sh install_service.sh
chown -R $USER:$USER .
```

### Проблемы с Docker
```bash
# Очистка Docker
docker system prune -a

# Пересборка образа
docker build --no-cache -t trading-bot .
```

### Проблемы с systemd
```bash
# Перезагрузка конфигурации
sudo systemctl daemon-reload

# Перезапуск сервиса
sudo systemctl restart trading-bot

# Проверка конфигурации
sudo systemd-analyze verify trading-bot.service
```

## 📈 Производительность

### Оптимизация для продакшена
1. Используйте systemd сервис для автозапуска
2. Настройте logrotate для управления логами
3. Используйте Docker для изоляции
4. Настройте мониторинг с Prometheus/Grafana
5. Используйте Redis для кэширования

### Мониторинг ресурсов
```bash
# Использование CPU и памяти
htop

# Использование диска
df -h

# Сетевые соединения
netstat -tulpn
```

## 🔒 Безопасность

### Рекомендации
1. Запускайте бота под отдельным пользователем
2. Ограничьте сетевой доступ
3. Используйте HTTPS для веб-интерфейса
4. Регулярно обновляйте зависимости
5. Мониторьте логи на предмет подозрительной активности

### Настройка firewall
```bash
# UFW
sudo ufw allow 8080/tcp
sudo ufw enable

# iptables
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
```

## 📝 Логирование

### Уровни логирования
- `DEBUG` - Подробная отладочная информация
- `INFO` - Общая информация о работе
- `WARNING` - Предупреждения
- `ERROR` - Ошибки

### Настройка логирования
```python
# В config.py
LOG_LEVEL = "INFO"
LOG_FILE = "bot.log"
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5
```

## 🤝 Поддержка

При возникновении проблем:
1. Проверьте логи
2. Убедитесь в корректности конфигурации
3. Проверьте зависимости
4. Обратитесь к документации
5. Создайте issue в репозитории

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл LICENSE для подробностей.
























