import pandas as pd
import matplotlib.pyplot as plt
import os

# Загрузка данных
df = pd.read_parquet("data/raw/test.parquet")

# Обработка даты и месяца
df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"])
df["month"] = df["transaction_timestamp"].dt.to_period("M")

# Папки для вывода
os.makedirs("output_parquet/graphs", exist_ok=True)
os.makedirs("output_parquet/csv", exist_ok=True)

# Обработка каждого клиента
for card_id in df["card_id"].unique():
    filtered = df[df["card_id"] == card_id]
    if len(filtered) < 5:
        continue  # пропустить слишком малые выборки

    # --- Сводки ---
    grouped_mcc = (
        filtered.groupby("mcc_category")["transaction_amount_kzt"]
        .sum().sort_values(ascending=False)
    )
    grouped_type = (
        filtered.groupby("transaction_type")["transaction_amount_kzt"]
        .sum().sort_values(ascending=False)
    )
    mcc_monthly = (
        filtered.groupby(["month", "mcc_category"])["transaction_amount_kzt"]
        .sum().unstack(fill_value=0)
    )
    type_monthly = (
        filtered.groupby(["month", "transaction_type"])["transaction_amount_kzt"]
        .sum().unstack(fill_value=0)
    )

    # --- Графики ---
    plt.figure(figsize=(14, 10))
    plt.suptitle(f"Анализ по клиенту card_id = {card_id}", fontsize=16)

    plt.subplot(2, 2, 1)
    grouped_mcc.plot(kind="bar", ax=plt.gca())
    plt.title("Суммы по MCC категориям")
    plt.ylabel("₸")
    plt.xticks(rotation=45, ha="right")

    plt.subplot(2, 2, 2)
    grouped_type.plot(kind="bar", color="orange", ax=plt.gca())
    plt.title("Суммы по типам транзакций")
    plt.ylabel("₸")
    plt.xticks(rotation=45, ha="right")

    plt.subplot(2, 2, 3)
    if mcc_monthly.shape[1] > 1:
        mcc_monthly.plot(ax=plt.gca(), marker="o")
    else:
        plt.text(0.5, 0.5, "Недостаточно данных", ha="center", va="center")
    plt.title("MCC категории по месяцам")
    plt.ylabel("₸")

    plt.subplot(2, 2, 4)
    if type_monthly.shape[1] > 1:
        type_monthly.plot(ax=plt.gca(), marker="o")
    else:
        plt.text(0.5, 0.5, "Недостаточно данных", ha="center", va="center")
    plt.title("Типы транзакций по месяцам")
    plt.ylabel("₸")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Сохранение ---
    plt.savefig(f"output_parquet/graphs/card_{card_id}.png")
    plt.close()

    filtered.to_csv(f"output_parquet/csv/card_{card_id}.csv", index=False)

    print(f"✅ Сохранено для card_id = {card_id}")
