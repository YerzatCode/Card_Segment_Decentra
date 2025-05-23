import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка метрик
df = pd.read_csv("customer_metrics.csv")

# === График 1: Распределение среднего чека (avg_amount)
plt.figure(figsize=(8, 5))
sns.histplot(df["avg_amount"], bins=50, kde=True)
plt.title("Распределение среднего чека (avg_amount)")
plt.xlabel("Средняя сумма транзакции")
plt.ylabel("Частота")
plt.grid(True)
plt.tight_layout()
plt.show()

# === График 2: Диаграмма рассеяния total_amount vs unique_mcc
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="unique_mcc", y="total_amount", alpha=0.7)
plt.title("Общая сумма vs Кол-во категорий")
plt.xlabel("Уникальные MCC")
plt.ylabel("Общая сумма транзакций")
plt.grid(True)
plt.tight_layout()
plt.show()

# === График 3: Ящик (Boxplot) по avg_days_between
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="avg_days_between")
plt.title("Интервал между транзакциями")
plt.xlabel("Среднее число дней между транзакциями")
plt.tight_layout()
plt.show()
