# websocket_monitor.py
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class WebSocketMonitor:
    """Simple WebSocket connection monitor for logging purposes."""
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}
        logger.info("WebSocket Monitor initialized.")
    
    def register_connection(self, connection_name: str):
        """Register a new WebSocket connection."""
        self.connections[connection_name] = {
            'registered_at': None,
            'message_count': 0
        }
        logger.info(f"Зарегистрировано соединение: {connection_name}")
    
    def record_message(self, connection_name: str):
        """Record a message received on a connection."""
        if connection_name in self.connections:
            self.connections[connection_name]['message_count'] += 1

# Global monitor instance
_monitor = None

def get_monitor() -> WebSocketMonitor:
    """Get the global WebSocket monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = WebSocketMonitor()
    return _monitor

