import nest_asyncio
nest_asyncio.apply()

import apscheduler.util as aps_util
import pytz

def patched_astimezone(timezone):
    if timezone is None:
        return None
    if hasattr(timezone, 'localize'):
        return timezone
    return pytz.UTC

aps_util.astimezone = patched_astimezone

import os
os.environ["PTB_DISABLE_JOB_QUEUE"] = "1"

import tzlocal
import pytz
tzlocal.get_localzone = lambda: pytz.timezone('UTC')

import pdfplumber
import logging
import asyncio
from telegram import Update, Document
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import camelot
import pandas as pd

# Замените 'YOUR_TELEGRAM_BOT_TOKEN' на токен вашего бота
TOKEN = "7638758608:AAF3awK3NRisz5dzCxfK2jMVC26W6D2DV-E"

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Отправь мне PDF-файл со спецификацией.")

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Получение файла от пользователя
    document = update.message.document
    file = await document.get_file()
    file_path = await file.download_to_drive()
    
    # Извлечение таблиц с помощью pdfplumber
    try:
        all_tables = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # Попытка извлечения таблиц со стандартными настройками
                tables = page.extract_tables()
                # Если стандартный метод не дал результатов или таблицы слишком маленькие, пробуем альтернативные настройки
                if not tables or all(len(table) < 2 for table in tables):
                    alt_tables = page.extract_tables(table_settings={
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "intersection_tolerance": 5
                    })
                    if alt_tables:
                        tables = alt_tables
                
                for table in tables:
                    if table is not None and len(table) > 1:
                        all_tables.append(table)
        
        if not all_tables:
            await update.message.reply_text("Не удалось извлечь таблицы из PDF с помощью pdfplumber.")
            return
    except Exception as e:
        await update.message.reply_text(f"Ошибка при извлечении таблиц с помощью pdfplumber: {e}")
        return
    
    # Преобразование списка таблиц в DataFrame
    dfs = []

    # Функция для создания уникальных имён столбцов
    def make_unique(cols):
        seen = {}
        unique_cols = []
        for col in cols:
            if col in seen:
                seen[col] += 1
                unique_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                unique_cols.append(col)
        return unique_cols

    for table in all_tables:
        header = None
        header_row_index = 0
        # Ищем строку, которая содержит подстроку 'поз' (без учета регистра)
        for i, row in enumerate(table):
            if any(cell and "поз" in cell.lower() for cell in row if cell is not None):
                header = row
                header_row_index = i
                break
        if header is None:
            # Если не найдено, используем первую строку как заголовок
            header = table[0]
            header_row_index = 0
        header = make_unique(header)
        data = table[header_row_index+1:]
        df_table = pd.DataFrame(data, columns=header)
        dfs.append(df_table)

    df = pd.concat(dfs, ignore_index=True)
        
    # Очистка заголовков от лишних пробелов
    df.columns = df.columns.astype(str).str.strip()
    
    # Фильтрация столбцов, имена которых начинаются с "None"
    df = df.loc[:, ~df.columns.str.startswith('None')]
    
    # Диагностическое сообщение с найденными столбцами
    await update.message.reply_text(f"Найденные столбцы: {df.columns.tolist()}")
    
    # Проверяем наличие столбца "Поз." или похожего (без учета регистра)
    if "Поз." not in df.columns:
        matched_cols = [col for col in df.columns if "поз" in col.lower()]
        if not matched_cols:
            await update.message.reply_text("В таблице не найден столбец, содержащий 'Поз'. Проверьте формат PDF.")
            return
        else:
            pos_col = matched_cols[0]
    else:
        pos_col = "Поз."
    
    # Группировка строк по столбцу pos_col
    groups = df.groupby(pos_col)
    
    # Создание Excel-файла с отдельными листами для каждой группы
    output_excel = "output.xlsx"
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        for group_name, group_df in groups:
            # Ограничение Excel: название листа не может превышать 31 символ
            sheet_name = str(group_name).strip()[:31]
            if not sheet_name:
                sheet_name = "Empty"
            group_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
    # Отправка Excel-файла пользователю
    await update.message.reply_document(document=open(output_excel, "rb"))

async def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.MimeType("application/pdf"), handle_pdf))
    
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())