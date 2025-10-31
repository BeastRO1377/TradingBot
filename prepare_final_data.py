# ======================================================================
# == prepare_final_data.py (ФИНАЛЬНАЯ ВЕРСИЯ С ЖЕСТКИМ ЛИМИТОМ)
# ======================================================================
import json
from pathlib import Path
import transformers
import logging

# --- НАСТРОЙКИ ---
SOURCE_FILE = Path("distillation_dataset.jsonl")
OUTPUT_DIR = Path("./lora_data_final")
TOKENIZER_PATH = "./Meta-Llama-3.1-8B-Instruct" # Используем токенизатор от Llama, он совместим
# [КЛЮЧЕВОЕ ИЗМЕНЕНИЕ] Жестко ограничиваем длину, чтобы избежать нехватки памяти
MAX_LENGTH = 1024 
# -----------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def prepare_and_truncate_data():
    logger.info("="*50)
    logger.info("--- Запуск финальной подготовки и ОБРЕЗКИ данных ---")
    logger.info("="*50)

    # ... (остальной код скрипта остается без изменений) ...
    # (Он будет здесь для полноты)
    if not SOURCE_FILE.exists():
        logger.error(f"❌ Исходный файл {SOURCE_FILE} не найден!")
        return

    if not Path(TOKENIZER_PATH).exists():
        logger.error(f"❌ Папка с токенизатором {TOKENIZER_PATH} не найдена!")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    train_path = OUTPUT_DIR / "train.jsonl"
    valid_path = OUTPUT_DIR / "valid.jsonl"

    logger.info("Загрузка токенизатора для измерения длины...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    logger.info(f"Обнаружено {len(data)} примеров в исходном файле.")
    
    processed_data = []
    truncated_count = 0
    for i, item in enumerate(data):
        try:
            user_content = item["messages"][0]["content"]
            assistant_content = item["messages"][1]["content"]

            full_text = (
                f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_content}<|eot_id|>"
            )

            token_ids = tokenizer.encode(full_text)
            if len(token_ids) > MAX_LENGTH:
                token_ids = token_ids[:MAX_LENGTH]
                truncated_count += 1
            
            final_text = tokenizer.decode(token_ids)
            processed_data.append({"text": final_text})

        except Exception as e:
            logger.warning(f"Пропуск строки {i+1} из-за ошибки: {e}")
            continue

    logger.info(f"Обработка завершена. Обрезано {truncated_count} слишком длинных примеров.")

    split_index = int(len(processed_data) * 0.95)
    train_data = processed_data[:split_index]
    valid_data = processed_data[split_index:]

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data: f.write(json.dumps(item) + "\n")
    logger.info(f"✅ Создан файл для обучения: {train_path.resolve()} ({len(train_data)} примеров)")

    with open(valid_path, "w", encoding="utf-8") as f:
        for item in valid_data: f.write(json.dumps(item) + "\n")
    logger.info(f"✅ Создан файл для валидации: {valid_path.resolve()} ({len(valid_data)} примеров)")
    
    print("\n" + "="*50)
    print("🎉 Данные полностью готовы для дообучения!")
    print("="*50)

if __name__ == "__main__":
    prepare_and_truncate_data()