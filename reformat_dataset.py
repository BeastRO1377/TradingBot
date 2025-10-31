# ======================================================================
# == reformat_dataset.py (Финальная версия с УПРОЩЕНИЕМ промпта)
# ======================================================================
import json
import os
from pathlib import Path

# --- НАСТРОЙКИ ---
SOURCE_FILE = Path("distillation_dataset.jsonl")
OUTPUT_DIR = Path("./lora_data")
TRAIN_RATIO = 0.95 # 95% данных пойдут на обучение, 5% на проверку
# -----------------

def reformat_and_simplify():
    print("="*50 + "\n--- Запуск финальной подготовки данных ---\n" + "="*50)

    if not SOURCE_FILE.exists():
        print(f"❌ ОШИБКА: Исходный файл {SOURCE_FILE} не найден!")
        return

    # Создаем папку для результатов
    OUTPUT_DIR.mkdir(exist_ok=True)
    train_path = OUTPUT_DIR / "train.jsonl"
    valid_path = OUTPUT_DIR / "valid.jsonl"

    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    print(f"Обнаружено {len(data)} примеров в исходном файле.")
    
    reformatted_data = []
    for item in data:
        try:
            # Извлекаем СУТЬ из старого формата
            user_content_json = json.loads(item["messages"][0]["content"].replace("Анализ сигнала: ", ""))
            assistant_content = item["messages"][1]["content"]

            # --- [КЛЮЧЕВОЕ ИЗМЕНЕНИЕ] ---
            # Создаем КОРОТКИЙ, но информативный промпт
            symbol = user_content_json.get("symbol", "N/A")
            side = user_content_json.get("side", "N/A")
            source = user_content_json.get("logic", "N/A") # В старом файле это 'logic'
            user_prompt = f"Проанализируй сигнал {side.upper()} для {symbol} от источника {source}."
            # ---------------------------

            # Собираем одну большую строку в формате Llama 3
            formatted_text = (
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{assistant_content}<|eot_id|>"
            )
            reformatted_data.append({"text": formatted_text})
        except Exception:
            continue # Пропускаем строки с ошибками

    # Разделяем данные
    split_index = int(len(reformatted_data) * 0.95)
    train_data = reformatted_data[:split_index]
    valid_data = reformatted_data[split_index:]

    # Сохраняем файлы
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data: f.write(json.dumps(item) + "\n")
    print(f"✅ Создан файл для обучения: {train_path.resolve()} ({len(train_data)} примеров)")

    with open(valid_path, "w", encoding="utf-8") as f:
        for item in valid_data: f.write(json.dumps(item) + "\n")
    print(f"✅ Создан файл для валидации: {valid_path.resolve()} ({len(valid_data)} примеров)")
    
    print("\n" + "="*50)
    print("🎉 Данные полностью готовы для дообучения!")
    print("="*50)

if __name__ == "__main__":
    reformat_and_simplify()