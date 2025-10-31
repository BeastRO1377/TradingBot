# websocket_monitor.py
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class WebSocketMonitor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WebSocketMonitor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.connections = {}
        self._initialized = True
        logger.info("WebSocket Monitor initialized.")

    def register_connection(self, name: str):
        self.connections[name] = {
            "status": "connected",
            "last_message_ts": time.time(),
            "connect_ts": time.time(),
            "message_count": 0,
            "error_count": 0,
            "reconnect_count": 0,
        }
        logger.info(f"Зарегистрировано соединение: {name}")

    def record_message(self, name: str):
        if name in self.connections:
            self.connections[name]["last_message_ts"] = time.time()
            self.connections[name]["message_count"] += 1

    def record_error(self, name: str, error_msg: str):
        if name in self.connections:
            self.connections[name]["status"] = "error"
            self.connections[name]["error_count"] += 1
        # Логируем только если есть сообщение об ошибке
        if error_msg:
            logger.error(f"Ошибка WebSocket [{name}]: {error_msg}")

    def record_reconnect(self, name: str):
        if name in self.connections:
            self.connections[name]["status"] = "reconnecting"
            self.connections[name]["reconnect_count"] += 1
        logger.warning(f"Переподключение WebSocket [{name}]...")

    def get_status(self) -> dict:
        return self.connections

# Глобальный экземпляр-одиночка (singleton)
_monitor_instance = WebSocketMonitor()

def get_monitor() -> WebSocketMonitor:
    """Возвращает единственный экземпляр монитора."""
    return _monitor_instance