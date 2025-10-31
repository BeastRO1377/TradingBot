import pandas as pd
import json

# Пороги для определения "хорошей" и "плохой" сделки
PROFIT_THRESHOLD = 1.0  # Сделка успешна, если ROI > 1%
LOSS_THRESHOLD = -1.0 # Сделка провальна, если ROI < -1%

df = pd.read_csv("finetune_log.csv")
print(f"Загружено {len(df)} записей для анализа.")

dataset = []
for index, row in df.iterrows():
    pnl = float(row["pnl_pct"])
    prompt = row["prompt"]
    
    ideal_action = None
    if pnl > PROFIT_THRESHOLD:
        ideal_action = "EXECUTE"
    elif pnl < LOSS_THRESHOLD:
        ideal_action = "REJECT"
    else:
        # Игнорируем сделки с нейтральным результатом, чтобы модель училась на явных примерах
        continue

    # Формируем идеальный JSON-ответ, которому мы хотим научить модель
    ideal_response = {
        "confidence_score": 1.0 if ideal_action == "EXECUTE" else 0.0,
        "justification": f"Это была {'прибыльная' if pnl > 0 else 'убыточная'} сделка, поэтому правильное действие - {ideal_action}.",
        "action": ideal_action
    }
    
    # Создаем запись в формате "вопрос-ответ"
    dataset.append({
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(ideal_response)}
        ]
    })

# Сохраняем в формате JSON Lines, который понимает Ollama
output_file = "finetune_dataset.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Готово! Создан файл {output_file} с {len(dataset)} обучающими примерами.")
