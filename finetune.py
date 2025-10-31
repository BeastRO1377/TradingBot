# ==============================================================================
# == finetune.py (ЕДИНЫЙ, САМОДОСТАТОЧНЫЙ СКРИПТ ДЛЯ ДООБУЧЕНИЯ)
# ==============================================================================
import argparse
import json
import logging
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm.utils import load, save_config
from tqdm import tqdm

# --- НАСТРОЙКИ ---
# Убедитесь, что пути соответствуют вашим файлам
MODEL_PATH = "./Meta-Llama-3.1-8B-Instruct"
DATA_PATH = "./training_data_final.jsonl" # Файл, который мы создали с помощью reformat_dataset.py
ADAPTER_PATH = "./mlx_adapters"
NUM_EPOCHS = 1 # Для ~500 примеров одной эпохи более чем достаточно
LEARNING_RATE = 1e-5
# -----------------

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class Dataset:
    def __init__(self, path: Path, tokenizer):
        if not path.exists():
            raise FileNotFoundError(f"Файл с датасетом не найден по пути: {path}")
        
        with open(path, "r") as fid:
            # Загружаем все данные сразу
            self.data = [json.loads(line) for line in fid]
        
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Токенизируем текст на лету
        text = self.data[idx]["text"]
        return self.tokenizer(
            text,
            return_tensors="np",
            return_attention_mask=False,
        )["input_ids"]

    def __len__(self):
        return len(self.data)

def main(args):
    logger.info("="*50)
    logger.info("--- Запуск финального скрипта дообучения ---")
    logger.info("="*50)

    # 1. Загрузка базовой модели
    logger.info(f"Загрузка базовой модели из: {args.model}")
    model, tokenizer = load(args.model)

    # 2. "Замораживаем" все слои модели
    model.freeze()
    # "Размораживаем" и заменяем только те слои, которые мы будем дообучать (LoRA)
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.set_lora()
    
    logger.info("Модель подготовлена для LoRA дообучения.")

    # 3. Загрузка датасета
    logger.info(f"Загрузка датасета из: {args.data}")
    train_dataset = Dataset(Path(args.data), tokenizer)

    # 4. Настройка обучения
    loss_and_grad_fn = nn.value_and_grad(model, lambda x, y: nn.losses.cross_entropy(x, y).mean())
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    # 5. Цикл дообучения
    for epoch in range(args.num_epochs):
        start_time = time.time()
        total_loss = 0
        
        pbar = tqdm(range(len(train_dataset)), desc=f"Эпоха {epoch + 1}/{args.num_epochs}")
        for i in pbar:
            sample = train_dataset[i]
            x = mx.array(sample[:, :-1])
            y = mx.array(sample[:, 1:])

            loss, grads = loss_and_grad_fn(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss / (i + 1):.5f}"})
        
        epoch_time = time.time() - start_time
        logger.info(f"Эпоха {epoch + 1} завершена за {epoch_time:.2f}с. Средняя ошибка: {total_loss / len(train_dataset):.5f}")

    # 6. Сохранение результата
    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(adapter_path / "adapters.safetensors"))
    # Сохраняем конфиг токенизатора, чтобы `fuse.py` его нашел
    tokenizer.save_pretrained(adapter_path)
    # Сохраняем конфиг LoRA
    with open(adapter_path / "adapter_config.json", "w") as f:
        json.dump({"lora_layers": -1}, f, indent=4)

    logger.info("="*50)
    logger.info("✅🎉 Дообучение успешно завершено!")
    logger.info(f"Адаптеры сохранены в: {adapter_path.resolve()}")
    logger.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт для LoRA дообучения.")
    # Добавляем аргументы по умолчанию, чтобы не писать их в командной строке
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Путь к базовой модели")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="Путь к файлу с данными для обучения")
    parser.add_argument("--adapter-path", type=str, default=ADAPTER_PATH, help="Путь для сохранения адаптеров")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS, help="Количество эпох обучения")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Скорость обучения")
    
    args = parser.parse_args()
    main(args)