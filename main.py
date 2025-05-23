import pandas as pd
import matplotlib.pyplot as plt
import os

# Загрузка данных
df = pd.read_csv("data/csv/transactions_processed.csv")

# Преобразуем дату и добавим месяц
df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"])
df["month"] = df["transaction_timestamp"].dt.to_period("M")

# Создаём папки для сохранения
os.makedirs("output/graphs", exist_ok=True)
os.makedirs("output/csv", exist_ok=True)

# Получаем всех клиентов
card_ids = df["card_id"].unique()

# Цикл по каждому клиенту
for card_id in card_ids:
    filtered = df[df["card_id"] == card_id]

    # Пропускаем если мало данных
    if len(filtered) < 5:
        continue

    # === Группировки ===
    grouped_mcc = filtered.groupby("mcc_category")["transaction_amount_kzt"].sum().sort_values(ascending=False)
    grouped_type = filtered.groupby("transaction_type")["transaction_amount_kzt"].sum().sort_values(ascending=False)
    mcc_monthly = filtered.groupby(["month", "mcc_category"])["transaction_amount_kzt"].sum().unstack(fill_value=0)
    type_monthly = filtered.groupby(["month", "transaction_type"])["transaction_amount_kzt"].sum().unstack(fill_value=0)

    # === Визуализация ===
    plt.figure(figsize=(14, 10))
    plt.suptitle(f"Анализ по клиенту card_id = {card_id}", fontsize=16)

    plt.subplot(2, 2, 1)
    grouped_mcc.plot(kind="bar", ax=plt.gca())
    plt.title("Суммы по MCC")
    plt.ylabel("₸")
    plt.xticks(rotation=45, ha="right")

    plt.subplot(2, 2, 2)
    grouped_type.plot(kind="bar", color="orange", ax=plt.gca())
    plt.title("Суммы по типам транзакций")
    plt.ylabel("₸")
    plt.xticks(rotation=45, ha="right")

    plt.subplot(2, 2, 3)
    mcc_monthly.plot(ax=plt.gca(), marker="o")
    plt.title("MCC по месяцам")
    plt.ylabel("₸")
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 4)
    type_monthly.plot(ax=plt.gca(), marker="o")
    plt.title("Типы транзакций по месяцам")
    plt.ylabel("₸")
    plt.xticks(rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Сохраняем как PNG
    graph_path = f"output/graphs/card_{card_id}.png"
    plt.savefig(graph_path)
    plt.close()

    # Сохраняем как CSV
    csv_path = f"output/csv/card_{card_id}.csv"
    filtered.to_csv(csv_path, index=False)

    print(f"✅ Сохранено для card_id = {card_id}: PNG и CSV")
