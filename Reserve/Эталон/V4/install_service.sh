#!/bin/bash
# Trading Bot Systemd Service Installer
# Скрипт для установки торгового бота как системного сервиса

set -euo pipefail

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Переменные
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="trading-bot"
SERVICE_USER="tradingbot"
INSTALL_DIR="/opt/trading-bot"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# Проверка прав root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "Этот скрипт должен быть запущен с правами root"
        log_info "Используйте: sudo $0"
        exit 1
    fi
}

# Создание пользователя
create_user() {
    log_info "Создание пользователя $SERVICE_USER..."
    
    if ! id "$SERVICE_USER" &>/dev/null; then
        useradd --system --shell /bin/false --home-dir "$INSTALL_DIR" --create-home "$SERVICE_USER"
        log_success "Пользователь $SERVICE_USER создан"
    else
        log_warning "Пользователь $SERVICE_USER уже существует"
    fi
}

# Установка файлов
install_files() {
    log_info "Установка файлов в $INSTALL_DIR..."
    
    # Создаем директорию установки
    mkdir -p "$INSTALL_DIR"
    
    # Копируем файлы проекта
    cp -r "$SCRIPT_DIR"/* "$INSTALL_DIR/"
    
    # Устанавливаем права доступа
    chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    chmod +x "$INSTALL_DIR/start_bot.sh"
    chmod +x "$INSTALL_DIR/launcher.py"
    
    # Создаем необходимые директории
    mkdir -p "$INSTALL_DIR/data" "$INSTALL_DIR/logs" "$INSTALL_DIR/models"
    chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/data" "$INSTALL_DIR/logs" "$INSTALL_DIR/models"
    
    log_success "Файлы установлены"
}

# Установка зависимостей
install_dependencies() {
    log_info "Установка зависимостей..."
    
    # Переключаемся на пользователя сервиса
    sudo -u "$SERVICE_USER" bash -c "
        cd '$INSTALL_DIR'
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    "
    
    log_success "Зависимости установлены"
}

# Установка systemd сервиса
install_service() {
    log_info "Установка systemd сервиса..."
    
    # Копируем файл сервиса
    cp "$SCRIPT_DIR/trading-bot.service" "$SERVICE_FILE"
    
    # Перезагружаем systemd
    systemctl daemon-reload
    
    # Включаем автозапуск
    systemctl enable "$SERVICE_NAME"
    
    log_success "Сервис установлен и включен"
}

# Настройка firewall (если нужно)
setup_firewall() {
    if command -v ufw &> /dev/null; then
        log_info "Настройка firewall..."
        ufw allow 8080/tcp comment "Trading Bot Web Interface"
        log_success "Firewall настроен"
    fi
}

# Создание конфигурации логирования
setup_logging() {
    log_info "Настройка логирования..."
    
    # Создаем конфигурацию logrotate
    cat > "/etc/logrotate.d/$SERVICE_NAME" << EOF
$INSTALL_DIR/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $SERVICE_USER $SERVICE_USER
    postrotate
        systemctl reload $SERVICE_NAME
    endscript
}
EOF
    
    log_success "Логирование настроено"
}

# Показать статус
show_status() {
    log_info "Статус сервиса:"
    systemctl status "$SERVICE_NAME" --no-pager
    
    log_info "Управление сервисом:"
    echo "  Запуск:    sudo systemctl start $SERVICE_NAME"
    echo "  Остановка: sudo systemctl stop $SERVICE_NAME"
    echo "  Перезапуск: sudo systemctl restart $SERVICE_NAME"
    echo "  Статус:    sudo systemctl status $SERVICE_NAME"
    echo "  Логи:      sudo journalctl -u $SERVICE_NAME -f"
}

# Удаление сервиса
uninstall() {
    log_info "Удаление сервиса..."
    
    # Останавливаем и отключаем сервис
    systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    systemctl disable "$SERVICE_NAME" 2>/dev/null || true
    
    # Удаляем файлы
    rm -f "$SERVICE_FILE"
    rm -f "/etc/logrotate.d/$SERVICE_NAME"
    
    # Перезагружаем systemd
    systemctl daemon-reload
    
    log_success "Сервис удален"
}

# Показать справку
show_help() {
    cat << EOF
Trading Bot Systemd Service Installer

Использование:
    $0 [ОПЦИИ]

Опции:
    -h, --help      Показать эту справку
    -u, --uninstall Удалить сервис
    -s, --status    Показать статус сервиса
    --start         Запустить сервис
    --stop          Остановить сервис
    --restart       Перезапустить сервис

Примеры:
    sudo $0                    # Установить сервис
    sudo $0 --uninstall        # Удалить сервис
    sudo $0 --status           # Показать статус
    sudo $0 --start            # Запустить сервис

EOF
}

# Основная функция
main() {
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -u|--uninstall)
            check_root
            uninstall
            exit 0
            ;;
        -s|--status)
            show_status
            exit 0
            ;;
        --start)
            check_root
            systemctl start "$SERVICE_NAME"
            show_status
            exit 0
            ;;
        --stop)
            check_root
            systemctl stop "$SERVICE_NAME"
            show_status
            exit 0
            ;;
        --restart)
            check_root
            systemctl restart "$SERVICE_NAME"
            show_status
            exit 0
            ;;
        "")
            # Установка
            check_root
            create_user
            install_files
            install_dependencies
            install_service
            setup_logging
            setup_firewall
            show_status
            log_success "Установка завершена!"
            ;;
        *)
            log_error "Неизвестная опция: $1"
            show_help
            exit 1
            ;;
    esac
}

# Запуск
main "$@"
























