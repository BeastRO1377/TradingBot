#!/bin/bash
# Trading Bot Startup Script
# Универсальный скрипт запуска торгового бота

set -euo pipefail

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Функция для отображения справки
show_help() {
    cat << EOF
Trading Bot Startup Script

Использование:
    $0 [ОПЦИИ] [РЕЖИМ]

Режимы:
    trading     Запуск в режиме реальной торговли (по умолчанию)
    demo        Запуск в демо режиме
    training    Обучение модели
    interactive Интерактивный режим

Опции:
    -h, --help              Показать эту справку
    -c, --config FILE       Путь к файлу конфигурации
    -d, --daemon            Запуск в фоновом режиме
    -l, --log-level LEVEL   Уровень логирования (DEBUG, INFO, WARNING, ERROR)
    -p, --port PORT         Порт для веб-интерфейса (если поддерживается)
    --check-deps            Только проверить зависимости
    --install-deps          Установить зависимости
    --update                Обновить бота из репозитория
    --stop                  Остановить все запущенные экземпляры
    --status                Показать статус бота
    --restart               Перезапустить бота

Примеры:
    $0                          # Запуск в режиме торговли
    $0 demo                     # Запуск в демо режиме
    $0 --daemon training        # Обучение в фоновом режиме
    $0 --check-deps             # Проверка зависимостей
    $0 --stop                   # Остановка бота
    $0 --restart                # Перезапуск бота

EOF
}

# Переменные
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
DAEMON_MODE=false
CONFIG_FILE=""
MODE="trading"
PID_FILE="$ROOT_DIR/bot.pid"
LOG_FILE="$ROOT_DIR/bot.log"

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--daemon)
            DAEMON_MODE=true
            shift
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -p|--port)
            export BOT_PORT="$2"
            shift 2
            ;;
        --check-deps)
            CHECK_DEPS=true
            shift
            ;;
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --update)
            UPDATE_BOT=true
            shift
            ;;
        --stop)
            STOP_BOT=true
            shift
            ;;
        --status)
            STATUS_BOT=true
            shift
            ;;
        --restart)
            RESTART_BOT=true
            shift
            ;;
        trading|demo|training|interactive)
            MODE="$1"
            shift
            ;;
        *)
            log_error "Неизвестная опция: $1"
            show_help
            exit 1
            ;;
    esac
done

# Функция проверки зависимостей
check_dependencies() {
    log_info "Проверка зависимостей..."
    
    # Проверяем Python
    if ! command -v "$PYTHON_BIN" &> /dev/null; then
        log_error "Python не найден. Установите Python 3.8 или выше."
        exit 1
    fi
    
    # Проверяем версию Python
    PYTHON_VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
        log_error "Требуется Python 3.8 или выше. Текущая версия: $PYTHON_VERSION"
        exit 1
    fi
    
    # Проверяем наличие основных файлов
    local required_files=("main.py" "bot_core.py" "config.py" "requirements.txt")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$ROOT_DIR/$file" ]]; then
            log_error "Отсутствует обязательный файл: $file"
            exit 1
        fi
    done
    
    # Проверяем виртуальное окружение
    if [[ ! -d "$VENV_DIR" ]]; then
        log_warning "Виртуальное окружение не найдено. Создание..."
        create_venv
    fi
    
    # Проверяем установку зависимостей
    if [[ ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
        log_warning "Виртуальное окружение повреждено. Пересоздание..."
        create_venv
    fi
    
    log_success "Все зависимости проверены"
}

# Функция создания виртуального окружения
create_venv() {
    log_info "Создание виртуального окружения..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    
    # Активируем окружение
    source "$VENV_DIR/bin/activate"
    
    # Обновляем pip
    pip install --upgrade pip
    
    # Устанавливаем зависимости
    log_info "Установка зависимостей..."
    pip install -r "$ROOT_DIR/requirements.txt"
    
    log_success "Виртуальное окружение создано"
}

# Функция установки зависимостей
install_dependencies() {
    log_info "Установка зависимостей..."
    
    if [[ ! -d "$VENV_DIR" ]]; then
        create_venv
    else
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip
        pip install -r "$ROOT_DIR/requirements.txt"
    fi
    
    log_success "Зависимости установлены"
}

# Функция остановки бота
stop_bot() {
    log_info "Остановка бота..."
    
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid"
            sleep 5
            if kill -0 "$pid" 2>/dev/null; then
                log_warning "Принудительное завершение процесса..."
                kill -KILL "$pid"
            fi
            log_success "Бот остановлен"
        else
            log_warning "Процесс с PID $pid не найден"
        fi
        rm -f "$PID_FILE"
    else
        log_warning "PID файл не найден. Попытка найти процесс по имени..."
        local pids=$(pgrep -f "python.*main.py" || true)
        if [[ -n "$pids" ]]; then
            echo "$pids" | xargs kill -TERM
            log_success "Процессы остановлены"
        else
            log_warning "Активные процессы бота не найдены"
        fi
    fi
}

# Функция проверки статуса
check_status() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log_success "Бот запущен (PID: $pid)"
            return 0
        else
            log_warning "PID файл существует, но процесс не запущен"
            rm -f "$PID_FILE"
            return 1
        fi
    else
        log_warning "Бот не запущен"
        return 1
    fi
}

# Функция запуска бота
start_bot() {
    local mode="$1"
    
    # Проверяем, не запущен ли уже бот
    if check_status >/dev/null 2>&1; then
        log_warning "Бот уже запущен. Используйте --restart для перезапуска."
        return 1
    fi
    
    # Проверяем зависимости
    check_dependencies
    
    # Активируем виртуальное окружение
    source "$VENV_DIR/bin/activate"
    
    # Настраиваем переменные окружения
    export PYTHONPATH="$ROOT_DIR"
    export PYTHONUNBUFFERED=1
    export LOG_LEVEL="$LOG_LEVEL"
    
    # Загружаем конфигурацию, если указана
    if [[ -n "$CONFIG_FILE" ]]; then
        export BOT_CONFIG_FILE="$CONFIG_FILE"
    fi
    
    # Формируем команду запуска
    local cmd=("$VENV_DIR/bin/python" "$ROOT_DIR/launcher.py" "--mode" "$mode")
    
    if [[ -n "$CONFIG_FILE" ]]; then
        cmd+=("--config" "$CONFIG_FILE")
    fi
    
    log_info "Запуск бота в режиме: $mode"
    
    if [[ "$DAEMON_MODE" == true ]]; then
        # Запуск в фоновом режиме
        nohup "${cmd[@]}" > "$LOG_FILE" 2>&1 &
        local pid=$!
        echo "$pid" > "$PID_FILE"
        log_success "Бот запущен в фоновом режиме (PID: $pid)"
        log_info "Логи: $LOG_FILE"
    else
        # Запуск в интерактивном режиме
        exec "${cmd[@]}"
    fi
}

# Функция перезапуска
restart_bot() {
    log_info "Перезапуск бота..."
    stop_bot
    sleep 2
    start_bot "$MODE"
}

# Функция обновления
update_bot() {
    log_info "Обновление бота..."
    
    # Здесь можно добавить логику обновления из Git репозитория
    # git pull origin main
    
    log_warning "Функция обновления не реализована"
}

# Основная логика
main() {
    # Обработка специальных команд
    if [[ "${CHECK_DEPS:-false}" == true ]]; then
        check_dependencies
        exit 0
    fi
    
    if [[ "${INSTALL_DEPS:-false}" == true ]]; then
        install_dependencies
        exit 0
    fi
    
    if [[ "${STOP_BOT:-false}" == true ]]; then
        stop_bot
        exit 0
    fi
    
    if [[ "${STATUS_BOT:-false}" == true ]]; then
        check_status
        exit $?
    fi
    
    if [[ "${RESTART_BOT:-false}" == true ]]; then
        restart_bot
        exit 0
    fi
    
    if [[ "${UPDATE_BOT:-false}" == true ]]; then
        update_bot
        exit 0
    fi
    
    # Обычный запуск
    start_bot "$MODE"
}

# Обработка сигналов
trap 'log_info "Получен сигнал завершения..."; stop_bot; exit 0' SIGINT SIGTERM

# Запуск
main "$@"
























