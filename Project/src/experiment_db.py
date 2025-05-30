import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "results.db" # Путь к файлу базы данных SQLite

def init_db():
    """Инициализирует базу данных: создает таблицу 'results' если она не существует."""
    conn = sqlite3.connect(DB_PATH) # Устанавливаем соединение с базой данных
    cursor = conn.cursor() # Создаем объект курсора для выполнения команд SQL
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results ( # Создать таблицу results, если она не существует
            id INTEGER PRIMARY KEY AUTOINCREMENT, # Уникальный идентификатор эксперимента (автоинкремент)
            date TEXT, # Дата и время проведения эксперимента (текстовое представление)
            real_data_path TEXT, # Путь к файлу реального изображения (если использовалось)
            gen_params TEXT, # Параметры генерации синтетического изображения (если использовалось, например, число клеток)
            method1_result INTEGER, # Результат анализа методом 1 (например, ML)
            method2_result INTEGER, # Результат анализа методом 2 (например, Clustering)
            method3_result INTEGER # Результат анализа методом 3 (например, CNN)
        )
    """)
    conn.commit() # Применяем изменения (создание таблицы)
    conn.close() # Закрываем соединение с базой данных

def save_experiment(date, real_data_path, gen_params, method1, method2, method3):
    """Сохраняет результаты одного эксперимента в таблицу 'results'."""
    conn = sqlite3.connect(DB_PATH) # Устанавливаем соединение с базой данных
    cursor = conn.cursor() # Создаем объект курсора
    cursor.execute("""
        INSERT INTO results (date, real_data_path, gen_params, method1_result, method2_result, method3_result)
        VALUES (?, ?, ?, ?, ?, ?) # Вставляем новую запись с параметрами
    """, (date, real_data_path, gen_params, method1, method2, method3)) # Передаем значения параметров
    conn.commit() # Сохраняем изменения
    conn.close() # Закрываем соединение

def load_results():
    """Загружает все записи из таблицы 'results' и возвращает их в виде pandas DataFrame."""
    conn = sqlite3.connect(DB_PATH) # Устанавливаем соединение
    df = pd.read_sql_query("SELECT * FROM results", conn) # Выполняем SQL-запрос и загружаем результат в DataFrame
    conn.close() # Закрываем соединение
    return df # Возвращаем DataFrame

def load_experiment_by_id(exp_id):
    """Загружает одну запись эксперимента по её уникальному идентификатору (id)."""
    conn = sqlite3.connect(DB_PATH) # Устанавливаем соединение
    df = pd.read_sql_query("SELECT * FROM results WHERE id = ?", conn, params=(exp_id,)) # Выполняем запрос с фильтром по id
    conn.close() # Закрываем соединение
    return df # Возвращаем DataFrame (будет содержать одну строку или быть пустым)

if __name__ == "__main__":
    """Пример использования функций модуля при запуске как основного скрипта."""
    init_db() # Инициализируем базу данных
    # Пример добавления эксперимента
    save_experiment(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Текущая дата и время
        real_data_path="path/to/image.jpg", # Пример пути
        gen_params="{'cells': 20, 'noise': 0.1}", # Пример параметров генерации
        method1=18, # Пример результата метода 1
        method2=20, # Пример результата метода 2
        method3=19 # Пример результата метода 3
    )
    # Пример вывода всех экспериментов из БД
    print(load_results())
