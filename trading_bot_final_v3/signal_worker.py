# signal_worker.py
from __future__ import annotations

import asyncio
import logging
import json
import re
from multiprocessing import Process, Queue
from typing import Any, Optional, Dict, List

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None  # чтобы модуль импортировался даже без openai


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [SignalWorker] %(message)s')
logger = logging.getLogger(__name__)


def safe_parse_json(text: Optional[str], default: Any = None) -> Any:
    if text is None:
        return default
    s = text.strip()
    # вырезаем возможные тройные кавычки с маркировкой ```json
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    # пробуем как есть
    try:
        return json.loads(s)
    except Exception:
        pass
    # пробуем найти самый внешний объект
    try:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end + 1])
    except Exception:
        pass
    return default


class AILogicWorker:
    """
    Асинхронный воркер, который:
    - читает задачи из task_queue (dict: {"candidate": {...}, "prompt": "..."}),
    - опрашивает LLM (OpenAI-совместимый endpoint, например Ollama /v1),
    - кладёт решение в order_queue.
    """
    def __init__(self, task_queue: Queue, order_queue: Queue, user_config: Dict[str, Any]):
        self.task_queue = task_queue
        self.order_queue = order_queue
        self.user_config = user_config or {}
        self.is_running = True

        self.ai_primary_model: str = self.user_config.get("ai_primary_model", "trading-llama")
        self.ollama_primary_openai: str = self.user_config.get("ollama_primary_openai", "http://localhost:11434/v1")
        self.ai_timeout_sec: float = float(self.user_config.get("ai_timeout_sec", 60.0))

    async def _ask_ollama_json(self, model: str, messages: List[dict], timeout_s: float) -> Dict[str, Any]:
        if AsyncOpenAI is None:
            logger.error("AsyncOpenAI не установлен. Верну REJECT.")
            return {"action": "REJECT", "justification": "openai package missing"}

        client = AsyncOpenAI(base_url=self.ollama_primary_openai, api_key="ollama")
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.2
                ),
                timeout=timeout_s,
            )
            # Важно: корректный доступ к первому choice
            raw = (resp.choices[0].message.content or "").strip()
            return safe_parse_json(raw, default={"action": "REJECT", "justification": "bad json"})
        except asyncio.TimeoutError:
            logger.error(f"Таймаут при запросе к модели {model} за {timeout_s}с.")
            return {"action": "REJECT", "justification": f"AI Timeout ({timeout_s}s)"}
        except Exception as e:
            logger.error(f"Ошибка при запросе к модели {model}: {e}", exc_info=True)
            return {"action": "REJECT", "justification": f"AI Error: {e}"}

    async def run(self):
        logger.info("Процесс-аналитик запущен и ожидает сигналы.")
        while self.is_running:
            try:
                if not self.task_queue.empty():
                    data_packet = self.task_queue.get_nowait()

                    candidate = data_packet.get("candidate")
                    prompt = data_packet.get("prompt")

                    if not (candidate and prompt):
                        await asyncio.sleep(0.01)
                        continue

                    logger.info(f"Получен сигнал для анализа: {candidate.get('symbol')}/{candidate.get('side')}")
                    messages = [{"role": "user", "content": prompt}]

                    ai_response = await self._ask_ollama_json(
                        model=self.ai_primary_model,
                        messages=messages,
                        timeout_s=self.ai_timeout_sec
                    )

                    if ai_response.get("action") == "EXECUTE":
                        order_command = {
                            "action": "EXECUTE_ORDER",
                            "symbol": candidate["symbol"],
                            "side": candidate["side"],
                            "source": candidate.get("source", "unknown"),
                            "volume_usdt": candidate.get("volume_usdt"),
                            "justification": ai_response.get("justification")
                        }
                        self.order_queue.put_nowait(order_command)
                    else:
                        self.order_queue.put_nowait({
                            "action": "REJECTED_BY_AI",
                            "symbol": candidate.get("symbol"),
                            "side": candidate.get("side"),
                            "justification": ai_response.get("justification", "no reason")
                        })
                else:
                    await asyncio.sleep(0.05)
            except Exception as e:
                logger.error(f"Ошибка в цикле воркера: {e}", exc_info=True)
                await asyncio.sleep(0.5)

    async def stop(self):
        self.is_running = False
        logger.info("Процесс-аналитик останавливается.")


def _worker_entry(task_q: Queue, order_q: Queue, user_cfg: Dict[str, Any]):
    """
    Точка входа дочернего процесса. Создаёт event loop и запускает воркер.
    Отдельный процесс предотвращает блокировки GIL и keeps I/O отдельным.
    """
    async def _run():
        worker = AILogicWorker(task_q, order_q, user_cfg)
        try:
            await worker.run()
        finally:
            await worker.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Фатальная ошибка воркера: {e}", exc_info=True)


def start_worker_process(user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Создаёт очереди и фоновый процесс-аналитик.
    Возвращает словарь с дескрипторами: process, task_queue, order_queue.
    Ничего не стартует при импортe; процесс поднимается только при вызове этой функции.
    """
    task_queue: Queue = Queue()
    order_queue: Queue = Queue()

    proc = Process(
        target=_worker_entry,
        args=(task_queue, order_queue, dict(user_config or {})),
        daemon=True
    )
    proc.start()
    logger.info(f"AI воркер запущен (pid={proc.pid}).")

    return {
        "process": proc,
        "task_queue": task_queue,
        "order_queue": order_queue
    }