import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt

# Папка с CSV-файлами по card_id
csv_folder = "output_parquet/csv"

top_mcc_per_client = []

for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        path = os.path.join(csv_folder, file)
        df = pd.read_csv(path)

        if "mcc_category" not in df.columns or df.empty:
            continue

        # Группируем по mcc_category — ищем где больше всего потратил
        top_mcc = (
            df.groupby("mcc_category")["transaction_amount_kzt"]
            .sum()
            .sort_values(ascending=False)
            .head(1)
            .index[0]
        )

        top_mcc_per_client.append(top_mcc)

# Подсчёт частоты по категориям
mcc_counts = Counter(top_mcc_per_client)
mcc_df = pd.DataFrame.from_dict(mcc_counts, orient="index", columns=["count"]).sort_values("count", ascending=False)

# === Визуализация
plt.figure(figsize=(10, 6))
mcc_df.head(15).plot(kind="bar", legend=False)
plt.title("🔝 Самые популярные MCC категории (где клиент тратит больше всего)")
plt.ylabel("Количество клиентов")
plt.xlabel("Категория MCC")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
