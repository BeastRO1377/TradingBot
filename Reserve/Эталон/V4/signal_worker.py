# signal_worker.py
import asyncio
import time
import logging
import json
import re
from multiprocessing import Queue
from typing import Dict, Any
from openai import AsyncOpenAI

# Настраиваем логирование для воркера
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [SignalWorker] %(message)s')

def safe_parse_json(text: str | None, default: Any = None) -> Any:
    if text is None: return default
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try: return json.loads(s)
    except: pass
    try:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end+1])
    except: pass
    return default

class AILogicWorker:
    def __init__(self, task_queue: Queue, order_queue: Queue, user_config: Dict[str, Any]):
        self.task_queue = task_queue
        self.order_queue = order_queue
        self.user_config = user_config
        self.is_running = True
        
        # Получаем настройки AI из переданного конфига, с фоллбэком на глобальные
        self.ai_primary_model = self.user_config.get("ai_primary_model", "trading-llama")
        self.ollama_primary_openai = self.user_config.get("ollama_primary_openai", "http://localhost:11434/v1")
        self.ai_timeout_sec = self.user_config.get("ai_timeout_sec", 60.0)

    async def _ask_ollama_json(self, model: str, messages: list[dict], timeout_s: float):
            client = AsyncOpenAI(base_url=self.ollama_primary_openai, api_key="ollama")
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(model=model, messages=messages, response_format={"type": "json_object"}, temperature=0.2),
                    timeout=timeout_s,
                )
                # --- ИСПРАВЛЕННАЯ СТРОКА ---
                # Правильно обращаемся к ПЕРВОМУ элементу списка 'choices'
                raw = (resp.choices[0].message.content or "").strip()
                # ---------------------------
                return safe_parse_json(raw, default={"action": "REJECT", "justification": "bad json"})
            except asyncio.TimeoutError:
                logging.error(f"Таймаут при запросе к модели {model} за {timeout_s}с.")
                return {"action": "REJECT", "justification": f"AI Timeout ({timeout_s}s)"}
            except Exception as e:
                logging.error(f"Ошибка при запросе к модели {model}: {e}", exc_info=True)
                return {"action": "REJECT", "justification": f"AI Error: {e}"}


    async def run(self):
        logging.info("Процесс-аналитик запущен и ожидает сигналы.")
        while self.is_running:
            try:
                if not self.task_queue.empty():
                    data_packet = self.task_queue.get_nowait()
                    
                    candidate = data_packet.get("candidate")
                    prompt = data_packet.get("prompt")
                    
                    if not (candidate and prompt): continue

                    logging.info(f"Получен сигнал для анализа: {candidate.get('symbol')}/{candidate.get('side')}")
                    messages = [{"role": "user", "content": prompt}]
                    
                    ai_response = await self._ask_ollama_json(
                        model=self.ai_primary_model,
                        messages=messages,
                        timeout_s=self.ai_timeout_sec
                    )
                    
                    if ai_response.get("action") == "EXECUTE":
                        order_command = {
                            "action": "EXECUTE_ORDER",
                            "symbol": candidate["symbol"], "side": candidate["side"],
                            "source": candidate["source"], "volume_usdt": candidate.get("volume_usdt"),
                            "justification": ai_response.get("justification")
                        }
                        self.order_queue.put(order_command)
                        logging.info(f"Сигнал {candidate['symbol']} ОДОБРЕН. Приказ отправлен на исполнение.")
                    else:
                        logging.info(f"Сигнал {candidate['symbol']} ОТКЛОНЕН AI. Причина: {ai_response.get('justification')}")

                await asyncio.sleep(0.1)
            except Exception as e:
                logging.error(f"Критическая ошибка в цикле воркера: {e}", exc_info=True)
                await asyncio.sleep(5)

def start_worker_process(task_q: Queue, order_q: Queue, config_dict: Dict[str, Any]):
    """Точка входа для нового процесса."""
    try:
        worker = AILogicWorker(task_q, order_q, config_dict)
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        logging.info("Процесс-аналитик остановлен.")
    except Exception as e:
        logging.critical(f"Процесс-аналитик упал с критической ошибкой: {e}", exc_info=True)