# Trading Bot Makefile
# Удобные команды для управления торговым ботом

.PHONY: help install run run-demo run-training stop status logs clean docker-build docker-run docker-stop test lint format

# Переменные
PYTHON := python3
VENV_DIR := .venv
PIP := $(VENV_DIR)/bin/pip
PYTHON_VENV := $(VENV_DIR)/bin/python
DOCKER_IMAGE := trading-bot
DOCKER_CONTAINER := trading-bot-container

# Цвета для вывода
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Справка
help: ## Показать справку
	@echo "$(GREEN)Trading Bot Management$(NC)"
	@echo "========================"
	@echo ""
	@echo "$(YELLOW)Основные команды:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Установка
install: ## Установить зависимости и настроить окружение
	@echo "$(YELLOW)Установка зависимостей...$(NC)"
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Зависимости установлены$(NC)"

# Запуск
run: ## Запустить бота в режиме торговли
	@echo "$(YELLOW)Запуск бота в режиме торговли...$(NC)"
	$(PYTHON_VENV) launcher.py --mode trading

run-demo: ## Запустить бота в демо режиме
	@echo "$(YELLOW)Запуск бота в демо режиме...$(NC)"
	$(PYTHON_VENV) launcher.py --mode demo

run-training: ## Запустить обучение модели
	@echo "$(YELLOW)Запуск обучения модели...$(NC)"
	$(PYTHON_VENV) launcher.py --mode training

run-interactive: ## Запустить в интерактивном режиме
	@echo "$(YELLOW)Запуск в интерактивном режиме...$(NC)"
	$(PYTHON_VENV) launcher.py --interactive

# Управление
stop: ## Остановить бота
	@echo "$(YELLOW)Остановка бота...$(NC)"
	@if [ -f bot.pid ]; then \
		kill $$(cat bot.pid) 2>/dev/null || true; \
		rm -f bot.pid; \
		echo "$(GREEN)Бот остановлен$(NC)"; \
	else \
		echo "$(RED)Бот не запущен$(NC)"; \
	fi

status: ## Показать статус бота
	@echo "$(YELLOW)Статус бота:$(NC)"
	@if [ -f bot.pid ]; then \
		if kill -0 $$(cat bot.pid) 2>/dev/null; then \
			echo "$(GREEN)Бот запущен (PID: $$(cat bot.pid))$(NC)"; \
		else \
			echo "$(RED)PID файл существует, но процесс не запущен$(NC)"; \
			rm -f bot.pid; \
		fi; \
	else \
		echo "$(RED)Бот не запущен$(NC)"; \
	fi

# Логи
logs: ## Показать логи бота
	@echo "$(YELLOW)Логи бота:$(NC)"
	@if [ -f bot.log ]; then \
		tail -f bot.log; \
	else \
		echo "$(RED)Файл логов не найден$(NC)"; \
	fi

logs-error: ## Показать только ошибки
	@echo "$(YELLOW)Ошибки в логах:$(NC)"
	@if [ -f bot.log ]; then \
		grep -i error bot.log | tail -20; \
	else \
		echo "$(RED)Файл логов не найден$(NC)"; \
	fi

# Очистка
clean: ## Очистить временные файлы
	@echo "$(YELLOW)Очистка временных файлов...$(NC)"
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.pyc
	rm -rf *.pyo
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	@echo "$(GREEN)Очистка завершена$(NC)"

clean-all: clean ## Полная очистка (включая виртуальное окружение)
	@echo "$(YELLOW)Полная очистка...$(NC)"
	rm -rf $(VENV_DIR)/
	rm -rf *.log
	rm -rf *.pid
	rm -rf data/
	rm -rf logs/
	@echo "$(GREEN)Полная очистка завершена$(NC)"

# Docker
docker-build: ## Собрать Docker образ
	@echo "$(YELLOW)Сборка Docker образа...$(NC)"
	docker build -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)Docker образ собран$(NC)"

docker-run: ## Запустить в Docker контейнере
	@echo "$(YELLOW)Запуск Docker контейнера...$(NC)"
	docker run -d \
		--name $(DOCKER_CONTAINER) \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/user_state.json:/app/user_state.json:ro \
		-p 8080:8080 \
		$(DOCKER_IMAGE)
	@echo "$(GREEN)Docker контейнер запущен$(NC)"

docker-stop: ## Остановить Docker контейнер
	@echo "$(YELLOW)Остановка Docker контейнера...$(NC)"
	docker stop $(DOCKER_CONTAINER) || true
	docker rm $(DOCKER_CONTAINER) || true
	@echo "$(GREEN)Docker контейнер остановлен$(NC)"

docker-logs: ## Показать логи Docker контейнера
	@echo "$(YELLOW)Логи Docker контейнера:$(NC)"
	docker logs -f $(DOCKER_CONTAINER)

docker-compose-up: ## Запустить через docker-compose
	@echo "$(YELLOW)Запуск через docker-compose...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Docker-compose запущен$(NC)"

docker-compose-down: ## Остановить docker-compose
	@echo "$(YELLOW)Остановка docker-compose...$(NC)"
	docker-compose down
	@echo "$(GREEN)Docker-compose остановлен$(NC)"

# Тестирование
test: ## Запустить тесты
	@echo "$(YELLOW)Запуск тестов...$(NC)"
	$(PYTHON_VENV) -m pytest tests/ -v

test-coverage: ## Запустить тесты с покрытием
	@echo "$(YELLOW)Запуск тестов с покрытием...$(NC)"
	$(PYTHON_VENV) -m pytest tests/ --cov=. --cov-report=html

# Линтинг и форматирование
lint: ## Проверить код линтерами
	@echo "$(YELLOW)Проверка кода...$(NC)"
	$(PYTHON_VENV) -m flake8 .
	$(PYTHON_VENV) -m mypy .
	$(PYTHON_VENV) -m black --check .

format: ## Форматировать код
	@echo "$(YELLOW)Форматирование кода...$(NC)"
	$(PYTHON_VENV) -m black .
	$(PYTHON_VENV) -m isort .

# Проверка зависимостей
check-deps: ## Проверить зависимости
	@echo "$(YELLOW)Проверка зависимостей...$(NC)"
	$(PYTHON_VENV) -m pip check
	$(PYTHON_VENV) -m safety check

update-deps: ## Обновить зависимости
	@echo "$(YELLOW)Обновление зависимостей...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt --upgrade

# Systemd сервис
install-service: ## Установить systemd сервис
	@echo "$(YELLOW)Установка systemd сервиса...$(NC)"
	sudo ./install_service.sh

uninstall-service: ## Удалить systemd сервис
	@echo "$(YELLOW)Удаление systemd сервиса...$(NC)"
	sudo ./install_service.sh --uninstall

start-service: ## Запустить systemd сервис
	@echo "$(YELLOW)Запуск systemd сервиса...$(NC)"
	sudo systemctl start trading-bot

stop-service: ## Остановить systemd сервис
	@echo "$(YELLOW)Остановка systemd сервиса...$(NC)"
	sudo systemctl stop trading-bot

restart-service: ## Перезапустить systemd сервис
	@echo "$(YELLOW)Перезапуск systemd сервиса...$(NC)"
	sudo systemctl restart trading-bot

status-service: ## Показать статус systemd сервиса
	@echo "$(YELLOW)Статус systemd сервиса:$(NC)"
	sudo systemctl status trading-bot

# Мониторинг
monitor: ## Показать мониторинг системы
	@echo "$(YELLOW)Мониторинг системы:$(NC)"
	@echo "CPU и память:"
	@top -bn1 | head -20
	@echo ""
	@echo "Дисковое пространство:"
	@df -h
	@echo ""
	@echo "Сетевые соединения:"
	@netstat -tulpn | grep :8080 || echo "Порт 8080 не используется"

# Резервное копирование
backup: ## Создать резервную копию
	@echo "$(YELLOW)Создание резервной копии...$(NC)"
	@mkdir -p backups
	@tar -czf backups/trading-bot-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		--exclude='.venv' \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		--exclude='.git' \
		--exclude='backups' \
		.
	@echo "$(GREEN)Резервная копия создана$(NC)"

# Восстановление
restore: ## Восстановить из резервной копии
	@echo "$(YELLOW)Доступные резервные копии:$(NC)"
	@ls -la backups/*.tar.gz 2>/dev/null || echo "Резервные копии не найдены"
	@echo ""
	@echo "Используйте: make restore-file FILE=backups/filename.tar.gz"

restore-file: ## Восстановить из конкретного файла
	@if [ -z "$(FILE)" ]; then \
		echo "$(RED)Укажите файл: make restore-file FILE=backups/filename.tar.gz$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Восстановление из $(FILE)...$(NC)"
	@tar -xzf $(FILE)
	@echo "$(GREEN)Восстановление завершено$(NC)"

# По умолчанию показываем справку
.DEFAULT_GOAL := help























