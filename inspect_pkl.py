import joblib  # <--- ИЗМЕНЕНИЕ: используем joblib вместо pickle
import pprint
import numpy as np

# Эти импорты нужны, чтобы joblib мог корректно восстановить объекты
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# Укажите имя вашего файла
filename = 'ml_policy_state.pkl'

try:
    # Используем joblib.load() вместо pickle.load()
    state = joblib.load(filename) # <--- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ

    print(f"--- Анализ модели из файла: {filename} ---")

    model = state.get('model')
    scaler = state.get('scaler')
    samples_seen = state.get('samples_seen', 'N/A')
    
    print(f"Всего обработано примеров: {samples_seen}\n")

    if model:
        print("--- Параметры модели (SGDClassifier) ---")
        
        # Проверяем, на каких классах обучена модель
        if hasattr(model, 'classes_'):
             print(f"Классы, которым обучена модель: {model.classes_}")
             if len(model.classes_) == 1:
                 print("\n!!! ВНИМАНИЕ: Модель обучена только на одном классе!")
                 print("Это объясняет, почему она всегда выдает одинаковую вероятность.\n")

        if hasattr(model, 'coef_'):
            print(f"Коэффициенты (веса) признаков: {model.coef_}")
            print(f"Свободный член (intercept/bias): {model.intercept_}")
        else:
            print("Модель еще не обучена (не имеет атрибута .coef_)")
            
        print("\nГиперпараметры модели:")
        pprint.pprint(model.get_params())
    else:
        print("Модель в файле не найдена.")

except FileNotFoundError:
    print(f"Ошибка: Файл '{filename}' не найден.")
except Exception as e:
    print(f"Произошла ошибка при чтении файла: {e}")