import pandas as pd

# Путь к файлу
file_path = 'data/raw/transactions.parquet'

# Загрузка данных
df = pd.read_parquet(file_path)

# Просмотр первых 10 строк
print("🔹 Первые строки:")
print(df.head(10))

# Информация о таблице
print("\n🔹 Структура таблицы:")
print(df.info())

# Основная статистика
print("\n🔹 Описательная статистика:")
print(df.describe(include='all'))  # Удален параметр datetime_is_numeric

# Количество уникальных клиентов
print("\n🔹 Количество уникальных card_id:")
print(df['card_id'].nunique())


