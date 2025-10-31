# ======================================================================
# == prepare_finetune_data.py (ЕДИНЫЙ СКРИПТ ПОДГОТОВКИ ДАННЫХ)
# ======================================================================
import json
from pathlib import Path
from mlx_lm.tuner.utils import save_dataset
import logging

# --- НАСТРОЙКИ ---
SOURCE_FILE = Path("distillation_dataset.jsonl")
OUTPUT_DIR = Path("./packed_data_for_finetune")
# -----------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def create_packed_dataset():
    logger.info("="*50)
    logger.info("--- Запуск финальной подготовки данных для дообучения ---")
    logger.info("="*50)

    if not SOURCE_FILE.exists():
        logger.error(f"❌ Исходный файл {SOURCE_FILE} не найден!")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Загружаем "сырые" данные
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    logger.info(f"Обнаружено {len(data)} примеров в исходном файле.")
    
    reformatted_data = []
    for item in data:
        try:
            # Преобразуем формат OpenAI в простой диалог
            user_content = item["messages"][0]["content"]
            assistant_content = item["messages"][1]["content"]
            reformatted_data.append({
                "conversations": [
                    {"from": "user", "value": user_content},
                    {"from": "assistant", "value": assistant_content},
                ]
            })
        except Exception:
            continue

    # Используем официальную утилиту для сохранения в правильном формате
    # Она сама разобьет данные на train/valid/test
    save_dataset(reformatted_data, OUTPUT_DIR)

    logger.info("\n" + "="*50)
    logger.info("🎉 Данные полностью готовы для дообучения!")
    logger.info(f"Упакованный датасет сохранен в папку: {OUTPUT_DIR.resolve()}")
    logger.info("="*50)

if __name__ == "__main__":
    create_packed_dataset()