import pandas as pd
import matplotlib.pyplot as plt

# Загрузи данные
df = pd.read_csv("data/csv/transactions_processed.csv")

# Укажи нужный card_id
target_card_id = 10001

# Преобразуем дату и выделим месяц
df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"])
df["month"] = df["transaction_timestamp"].dt.to_period("M")

# Отфильтруй по этому клиенту
filtered = df[df["card_id"] == target_card_id]

# === ГРАФИК 1: Суммы по категориям MCC ===
grouped_mcc = (
    filtered.groupby("mcc_category")["transaction_amount_kzt"]
    .sum()
    .sort_values(ascending=False)
)

# === ГРАФИК 2: Распределение по типу транзакций ===
grouped_type = (
    filtered.groupby("transaction_type")["transaction_amount_kzt"]
    .sum()
    .sort_values(ascending=False)
)

# === ГРАФИК 3: Суммы по MCC категориям по месяцам ===
mcc_monthly = (
    filtered.groupby(["month", "mcc_category"])["transaction_amount_kzt"]
    .sum()
    .unstack(fill_value=0)
)

# === ГРАФИК 4: Суммы по типам транзакций по месяцам ===
type_monthly = (
    filtered.groupby(["month", "transaction_type"])["transaction_amount_kzt"]
    .sum()
    .unstack(fill_value=0)
)

# === ВИЗУАЛИЗАЦИЯ ===
plt.figure(figsize=(14, 10))

# 1. Суммы по MCC
plt.subplot(2, 2, 1)
grouped_mcc.plot(kind="bar", ax=plt.gca())
plt.title(f"Суммы покупок по категориям (card_id = {target_card_id})")
plt.ylabel("Сумма, ₸")
plt.xlabel("Категория (MCC)")
plt.xticks(rotation=45, ha="right")

# 2. Типы транзакций
plt.subplot(2, 2, 2)
grouped_type.plot(kind="bar", color="orange", ax=plt.gca())
plt.title("Распределение по типу операций")
plt.ylabel("Сумма, ₸")
plt.xlabel("Тип транзакции")
plt.xticks(rotation=45, ha="right")

# 3. Динамика MCC по месяцам
plt.subplot(2, 2, 3)
mcc_monthly.plot(ax=plt.gca(), marker="o")
plt.title("Динамика по MCC категориям")
plt.ylabel("Сумма, ₸")
plt.xlabel("Месяц")
plt.xticks(rotation=45)

# 4. Динамика типов по месяцам
plt.subplot(2, 2, 4)
type_monthly.plot(ax=plt.gca(), marker="o")
plt.title("Динамика по типам транзакций")
plt.ylabel("Сумма, ₸")
plt.xlabel("Месяц")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
