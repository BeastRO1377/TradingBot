# generate_dataset.py
import asyncio
import pandas as pd
import os
from openai import AsyncOpenAI
import json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")
SIGNALS_CSV_PATH = "ml_best_entries.csv" # Убедитесь, что этот файл существует
OUTPUT_JSONL_PATH = "distillation_dataset.jsonl"
SIGNALS_TO_PROCESS = 4000 # На сколько сигналов хватит $20 (примерно)

async def generate_distilled_response(client, features):
    # Используем новый, обогащенный промпт
    prompt = f"""
    SYSTEM: Ты - элитный финансовый аналитик. Проанализируй сигнал и дай подробное обоснование.

    USER:
    Проведи глубокий анализ сигнала.
    - Контекст: {json.dumps(features, indent=2)}
    
    ЗАДАЧА: Дай развернутое объяснение, почему этот сигнал является хорошим или плохим. В конце вынеси вердикт "EXECUTE" или "REJECT".
    """
    
    response = await client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content

async def main():
    try:
        df = pd.read_csv(SIGNALS_CSV_PATH).head(SIGNALS_TO_PROCESS)
    except FileNotFoundError:
        print(f"Ошибка: Файл {SIGNALS_CSV_PATH} не найден. Сначала соберите данные с помощью бота.")
        return
        
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for index, row in df.iterrows():
            features = row.to_dict()
            print(f"Обработка сигнала {index+1}/{len(df)}...")
            
            try:
                ideal_response = await generate_distilled_response(client, features)
                
                # Формируем обучающий пример
                training_example = {
                    "messages": [
                        {"role": "user", "content": f"Анализ сигнала: {json.dumps(features)}"},
                        {"role": "assistant", "content": ideal_response}
                    ]
                }
                f.write(json.dumps(training_example, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Ошибка при обработке строки {index}: {e}")

if __name__ == "__main__":
    asyncio.run(main())