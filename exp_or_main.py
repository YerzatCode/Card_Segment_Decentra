import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Проверка наличия файлов
print("CSV-файлы в проекте:")
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".csv"):
            print(os.path.join(root, file))

# 2. Загрузка транзакций
df = pd.read_csv('data/processed/transactions_processed.csv')
print("Колонки в df:", df.columns.tolist())

# 3. Загрузка MCC-справочника
mcc_df = pd.read_excel('data/processed/mcc_codes_and_categories.xlsx')
print("Столбцы в mcc_df:", mcc_df.columns.tolist())

# 4. Словарь MCC → категория
mcc_dict = dict(zip(
    mcc_df['merchant_mcc'],
    mcc_df['mcc_category']
))

# 5. Маппинг и заполнение пропусков
df['mcc_name'] = df['merchant_mcc'].map(mcc_dict).fillna('Неизвестная категория')

# 6. Укажите здесь нужное card_id
card_id = 10000 # <-- замените на тот, который нужен

# 7. Фильтрация по карте
df_card = df[df['card_id'] == card_id]
if df_card.empty:
    raise ValueError(f"Нет транзакций для card_id={card_id}")

# 8. Группировка трат по категориям
#     Убедитесь, что имя колонки с суммами верно:
amount_col = 'transaction_amount_kzt'  # или 'amount', как у вас
if amount_col not in df_card.columns:
    raise KeyError(f"Колонка суммы '{amount_col}' не найдена в df")

spending_by_cat = (
    df_card
    .groupby('mcc_name')[amount_col]
    .sum()
    .sort_values(ascending=False)
)

# 9. Вывод результатов в консоль
print(f"\nТраты для карты {card_id} по категориям MCC:")
print(spending_by_cat.to_frame(name='Сумма трат').reset_index())

# 10. Построение bar-chart
plt.figure(figsize=(10, 6))
spending_by_cat.plot(kind='bar')
plt.title(f'Траты по категориям MCC для карты {card_id}', fontsize=14)
plt.ylabel('Сумма трат', fontsize=12)
plt.xlabel('Категория MCC', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
