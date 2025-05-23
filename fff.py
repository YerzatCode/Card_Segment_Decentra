import pandas as pd

# Загрузи твой parquet
df = pd.read_parquet("data/raw/test.parquet")

# Покажи список колонок
print("📋 Колонки в parquet:")
print(df.columns)

# Покажи первые строки данных
print("\n🔍 Пример данных:")
print(df.head())
