import os
import sys
from huggingface_hub import snapshot_download, HfFolder

# --- НАСТРОЙКИ ---
# [ФИНАЛЬНЫЙ ВЫБОР] Llama 3.3 (8B Instruct) - самая последняя и мощная
# модель, оптимальная для локальной работы на Apple Silicon.
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Папка, куда будет сохранена модель
LOCAL_DIR = "./Meta-Llama-3.1-8B-Instruct"
# -----------------

def main():
    print("="*50 + "\n--- Запуск умного загрузчика моделей ---\n" + "="*50)

    token = HfFolder.get_token()
    if token is None:
        print("\n❌ ОШИБКА: Вы не авторизованы в Hugging Face.")
        print("Пожалуйста, выполните в терминале: huggingface-cli login")
        sys.exit(1)
    else:
        print("\n✅ Авторизация в Hugging Face найдена.")

    print(f"\n[ШАГ 1] Создание папки: {os.path.abspath(LOCAL_DIR)}")
    os.makedirs(LOCAL_DIR, exist_ok=True)

    print(f"\n[ШАГ 2] Начинается скачивание модели {MODEL_ID}...")
    print("Будут загружены только необходимые файлы (~16 ГБ).")
    
    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            allow_patterns=["*.safetensors", "*.json", "tokenizer.model", "*.py", "*.md"],
            local_dir_use_symlinks=False, 
        )
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        print("Проверьте интернет и убедитесь, что приняли лицензию на странице модели.")
        sys.exit(1)

    print("\n" + "="*50)
    print("✅🎉 Скачивание успешно завершено!")
    print(f"Модель сохранена в: {os.path.abspath(LOCAL_DIR)}")
    print("Теперь можно переходить к дообучению.")
    print("="*50)

if __name__ == "__main__":
    main()