#!/usr/bin/env python3
"""
Скрипт для безопасного удаления дубликатов в коде TradingBot
"""

import os
import hashlib
import shutil
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def get_file_hash(filepath):
    """Вычисляет MD5 хеш файла"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        return None

def backup_file(filepath, backup_dir):
    """Создает резервную копию файла"""
    try:
        backup_path = backup_dir / filepath.name
        shutil.copy2(filepath, backup_path)
        return backup_path
    except Exception as e:
        print(f"❌ Ошибка при создании резервной копии {filepath}: {e}")
        return None

def find_exact_duplicates():
    """Находит точные дубликаты по хешу"""
    current_dir = Path.cwd()
    hash_groups = defaultdict(list)
    
    # Ищем только файлы MultiuserBot*
    for root, dirs, files in os.walk(current_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        for file in files:
            if file.startswith('MultiuserBot') and file.endswith('.py'):
                filepath = Path(root) / file
                file_hash = get_file_hash(filepath)
                if file_hash:
                    hash_groups[file_hash].append(filepath)
    
    # Фильтруем только группы с дубликатами
    duplicates = {hash_val: files for hash_val, files in hash_groups.items() if len(files) > 1}
    return duplicates

def cleanup_duplicates(dry_run=True, create_backup=True):
    """Удаляет дубликаты с возможностью отката"""
    print("🧹 Очистка дубликатов в коде TradingBot")
    print("=" * 60)
    
    if dry_run:
        print("🔍 РЕЖИМ ПРЕДВАРИТЕЛЬНОГО ПРОСМОТРА (файлы не будут удалены)")
    else:
        print("⚠️  РЕЖИМ РЕАЛЬНОГО УДАЛЕНИЯ")
    
    # Создаем директорию для резервных копий
    backup_dir = None
    if create_backup and not dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f"backup_duplicates_{timestamp}")
        backup_dir.mkdir(exist_ok=True)
        print(f"📁 Создана директория для резервных копий: {backup_dir}")
    
    # Находим дубликаты
    duplicates = find_exact_duplicates()
    
    if not duplicates:
        print("✅ Дубликатов не найдено")
        return
    
    print(f"\n📊 Найдено групп дубликатов: {len(duplicates)}")
    
    total_files_to_remove = 0
    total_size_to_save = 0
    cleanup_plan = []
    
    # Анализируем каждую группу дубликатов
    for file_hash, files in duplicates.items():
        print(f"\n🔄 Группа дубликатов (хеш: {file_hash[:8]}...):")
        
        # Сортируем файлы по дате модификации (оставляем самый новый)
        files_with_time = []
        for filepath in files:
            try:
                stat = os.stat(filepath)
                files_with_time.append((filepath, stat.st_mtime, stat.st_size))
            except:
                continue
        
        files_with_time.sort(key=lambda x: x[1], reverse=True)  # Сортировка по времени (новые первыми)
        
        # Первый файл (самый новый) оставляем, остальные удаляем
        keep_file = files_with_time[0]
        remove_files = files_with_time[1:]
        
        print(f"   ✅ Оставляем: {keep_file[0].name} (размер: {keep_file[2]} байт)")
        
        for filepath, mtime, size in remove_files:
            print(f"   🗑️  Удаляем: {filepath.name} (размер: {size} байт)")
            total_files_to_remove += 1
            total_size_to_save += size
            
            if not dry_run:
                # Создаем резервную копию
                if create_backup:
                    backup_path = backup_file(filepath, backup_dir)
                    if backup_path:
                        print(f"      💾 Резервная копия: {backup_path}")
                
                # Удаляем файл
                try:
                    os.remove(filepath)
                    print(f"      ✅ Файл удален")
                except Exception as e:
                    print(f"      ❌ Ошибка при удалении: {e}")
        
        cleanup_plan.append({
            'hash': file_hash,
            'keep': str(keep_file[0]),
            'remove': [str(f[0]) for f in remove_files],
            'size_saved': sum(f[2] for f in remove_files)
        })
    
    # Сохраняем план очистки
    if not dry_run:
        cleanup_log = {
            'timestamp': datetime.now().isoformat(),
            'total_files_removed': total_files_to_remove,
            'total_size_saved_bytes': total_size_to_save,
            'total_size_saved_kb': total_size_to_save / 1024,
            'backup_directory': str(backup_dir) if backup_dir else None,
            'cleanup_plan': cleanup_plan
        }
        
        cleanup_log_path = Path(f"cleanup_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(cleanup_log_path, 'w', encoding='utf-8') as f:
            json.dump(cleanup_log, f, indent=2, ensure_ascii=False)
        
        print(f"\n📝 Лог очистки сохранен: {cleanup_log_path}")
    
    # Итоговая статистика
    print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"   Файлов для удаления: {total_files_to_remove}")
    print(f"   Экономия места: {total_size_to_save / 1024:.1f} KB")
    
    if backup_dir:
        print(f"   Резервные копии: {backup_dir}")
        print(f"   💡 Для отката: скопируйте файлы из {backup_dir} обратно")
    
    if dry_run:
        print(f"\n🔍 Это был предварительный просмотр.")
        print(f"   Для реального удаления запустите: python3 {__file__} --execute")
    else:
        print(f"\n✅ Очистка завершена!")
        print(f"   🗑️  Удалено файлов: {total_files_to_remove}")
        print(f"   💾 Резервные копии сохранены в: {backup_dir}")

def main():
    import sys
    
    dry_run = True
    create_backup = True
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--execute':
            dry_run = False
            print("⚠️  ВНИМАНИЕ: Будет выполнено реальное удаление файлов!")
            response = input("Продолжить? (y/N): ")
            if response.lower() != 'y':
                print("❌ Операция отменена")
                return
        elif sys.argv[1] == '--no-backup':
            create_backup = False
        elif sys.argv[1] == '--help':
            print("Использование:")
            print("  python3 cleanup_duplicates.py          # Предварительный просмотр")
            print("  python3 cleanup_duplicates.py --execute # Реальное удаление")
            print("  python3 cleanup_duplicates.py --no-backup # Без резервных копий")
            return
    
    cleanup_duplicates(dry_run=dry_run, create_backup=create_backup)

if __name__ == "__main__":
    main()





























