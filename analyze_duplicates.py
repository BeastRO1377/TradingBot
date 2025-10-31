#!/usr/bin/env python3
"""
Скрипт для анализа дубликатов в коде TradingBot
"""

import os
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import difflib

def get_file_hash(filepath):
    """Вычисляет MD5 хеш файла"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Ошибка при чтении файла {filepath}: {e}")
        return None

def get_file_info(filepath):
    """Получает информацию о файле"""
    try:
        stat = os.stat(filepath)
        return {
            'size': stat.st_size,
            'mtime': stat.st_mtime,
            'path': str(filepath)
        }
    except Exception as e:
        print(f"Ошибка при получении информации о файле {filepath}: {e}")
        return None

def find_duplicates_by_hash(directory):
    """Находит дубликаты по хешу файлов"""
    hash_groups = defaultdict(list)
    
    for root, dirs, files in os.walk(directory):
        # Пропускаем .history и другие служебные директории
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                file_hash = get_file_hash(filepath)
                if file_hash:
                    hash_groups[file_hash].append(filepath)
    
    # Фильтруем только группы с дубликатами
    duplicates = {hash_val: files for hash_val, files in hash_groups.items() if len(files) > 1}
    return duplicates

def find_similar_files(directory, similarity_threshold=0.8):
    """Находит файлы с похожим содержимым"""
    py_files = []
    
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                py_files.append(filepath)
    
    similar_groups = []
    
    for i, file1 in enumerate(py_files):
        try:
            with open(file1, 'r', encoding='utf-8', errors='ignore') as f:
                content1 = f.read()
        except Exception:
            continue
            
        for file2 in py_files[i+1:]:
            try:
                with open(file2, 'r', encoding='utf-8', errors='ignore') as f:
                    content2 = f.read()
            except Exception:
                continue
            
            # Вычисляем схожесть
            similarity = difflib.SequenceMatcher(None, content1, content2).ratio()
            
            if similarity >= similarity_threshold:
                similar_groups.append({
                    'file1': str(file1),
                    'file2': str(file2),
                    'similarity': similarity
                })
    
    return similar_groups

def analyze_multiuserbot_files(directory):
    """Анализирует файлы MultiuserBot*"""
    multiuserbot_files = []
    
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        for file in files:
            if file.startswith('MultiuserBot') and file.endswith('.py'):
                filepath = Path(root) / file
                file_info = get_file_info(filepath)
                if file_info:
                    multiuserbot_files.append(file_info)
    
    # Группируем по размеру
    size_groups = defaultdict(list)
    for file_info in multiuserbot_files:
        size_groups[file_info['size']].append(file_info)
    
    return multiuserbot_files, size_groups

def main():
    print("🔍 Анализ дубликатов в коде TradingBot")
    print("=" * 50)
    
    current_dir = Path.cwd()
    
    # 1. Анализ файлов MultiuserBot*
    print("\n1. Анализ файлов MultiuserBot*:")
    multiuserbot_files, size_groups = analyze_multiuserbot_files(current_dir)
    
    print(f"   Найдено файлов: {len(multiuserbot_files)}")
    
    # Показываем группы по размеру
    for size, files in size_groups.items():
        if len(files) > 1:
            print(f"\n   Группа файлов размером {size} байт ({len(files)} файлов):")
            for file_info in files:
                print(f"     - {file_info['path']}")
    
    # 2. Поиск точных дубликатов по хешу
    print("\n2. Поиск точных дубликатов:")
    duplicates = find_duplicates_by_hash(current_dir)
    
    if duplicates:
        print(f"   Найдено групп дубликатов: {len(duplicates)}")
        total_duplicate_files = sum(len(files) for files in duplicates.values())
        print(f"   Общее количество дублированных файлов: {total_duplicate_files}")
        
        for file_hash, files in list(duplicates.items())[:5]:  # Показываем первые 5 групп
            print(f"\n   Группа дубликатов (хеш: {file_hash[:8]}...):")
            for filepath in files:
                print(f"     - {filepath}")
    else:
        print("   Точных дубликатов не найдено")
    
    # 3. Поиск похожих файлов
    print("\n3. Поиск похожих файлов (схожесть >= 80%):")
    similar_files = find_similar_files(current_dir, similarity_threshold=0.8)
    
    if similar_files:
        print(f"   Найдено пар похожих файлов: {len(similar_files)}")
        
        # Группируем похожие файлы
        similarity_groups = defaultdict(list)
        for pair in similar_files:
            similarity_groups[round(pair['similarity'], 2)].append(pair)
        
        for similarity in sorted(similarity_groups.keys(), reverse=True):
            pairs = similarity_groups[similarity]
            print(f"\n   Файлы со схожестью {similarity*100:.0f}% ({len(pairs)} пар):")
            for pair in pairs[:3]:  # Показываем первые 3 пары
                print(f"     - {pair['file1']}")
                print(f"       {pair['file2']}")
                print(f"       Схожесть: {pair['similarity']*100:.1f}%")
                print()
    else:
        print("   Похожих файлов не найдено")
    
    # 4. Статистика по размерам
    print("\n4. Статистика по размерам файлов:")
    sizes = [f['size'] for f in multiuserbot_files]
    if sizes:
        total_size = sum(sizes)
        avg_size = total_size / len(sizes)
        print(f"   Общий размер всех файлов: {total_size / 1024:.1f} KB")
        print(f"   Средний размер файла: {avg_size / 1024:.1f} KB")
        print(f"   Минимальный размер: {min(sizes) / 1024:.1f} KB")
        print(f"   Максимальный размер: {max(sizes) / 1024:.1f} KB")
    
    # 5. Рекомендации по очистке
    print("\n5. Рекомендации по очистке:")
    if duplicates:
        print("   ⚠️  Найдены точные дубликаты - можно безопасно удалить")
        print("   💡 Рекомендуется оставить только один файл из каждой группы дубликатов")
    
    if similar_files:
        print("   ⚠️  Найдены похожие файлы - требуется ручная проверка")
        print("   💡 Рекомендуется сравнить содержимое и объединить функциональность")
    
    print(f"\n   📁 Всего файлов MultiuserBot*: {len(multiuserbot_files)}")
    print("   🗑️  Рекомендуется провести рефакторинг и удалить дубликаты")

if __name__ == "__main__":
    main()





























