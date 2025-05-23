import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загрузка данных из разных источников
# Здесь можно подгрузить данные о клиентах, транзакциях, а также внешние данные (соцсети, обращения в колл-центр и т.д.)
df_client = pd.read_csv('data/processed/client_features.csv')
df_trans = pd.read_csv('data/processed/transactions_processed.csv')
# Если имеются и другие источники, например: df_external = pd.read_csv('data/processed/external_data.csv')

# Для простоты будем использовать объединение данных по card_id
df = pd.merge(df_client, df_trans, on='card_id', how='left')

# 2. Приведение дат к корректному формату
df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'], errors='coerce')
df['expiry_date'] = pd.to_datetime('20' + df['expiry_date'], format='%Y%m', errors='coerce')

# 3. Предобработка данных: заполнение пропусков
# Заполняем числовые признаки медианой, категориальные — значением 'Unknown'
num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna('Unknown')

# 4. Агрегация транзакционных данных по клиенту с расширенным набором признаков
def compute_avg_gap(x):
    # Если имеется несколько транзакций, вычисляем средний интервал между ними в днях
    if len(x) > 1:
        x = x.sort_values()
        # Разница между соседними транзакциями
        gap_days = x.diff().dropna().dt.days.mean()
        return gap_days
    else:
        return np.nan

agg_funcs = {
    # Суммарные показатели и статистики по суммам
    'transaction_amount_kzt': ['sum', 'mean', 'count', 'std'],
    # Количество уникальных типов транзакций (POS, ATM, P2P и т.д.)
    'transaction_type': pd.Series.nunique,
    # Количество уникальных торговых категорий (MCC)
    'merchant_mcc': pd.Series.nunique,
    # Количество уникальных городов проведения операций
    'merchant_city': pd.Series.nunique,
    # Средний интервал между транзакциями
    'transaction_timestamp': compute_avg_gap,
    # Доля транзакций, проведённых внутри страны (здесь KAZ — код для Казахстана)
    'acquirer_country_iso': lambda x: (x == 'KAZ').sum() / x.count() if x.count() > 0 else 0,
    # Разнообразие режимов ввода (Chip, Magstripe, Contactless и т.д.)
    'pos_entry_mode': pd.Series.nunique,
    # Разнообразие типов кошельков (ApplePay, GooglePay и пр.)
    'wallet_type': pd.Series.nunique
}

agg_df = df.groupby('card_id').agg(agg_funcs)
agg_df.columns = [
    'total_amount', 'avg_amount', 'transaction_count', 'std_amount',
    'unique_transaction_types', 'unique_merchant_mcc', 'unique_merchant_city',
    'avg_tx_gap_days', 'domestic_ratio', 'unique_pos_entry_mode', 'unique_wallet_type'
]
agg_df.reset_index(inplace=True)

# Заполняем пропуски полученных агрегатов (например, при единичных транзакциях avg_tx_gap_days)
agg_df.fillna(agg_df.median(), inplace=True)

# 5. Формирование целевой переменной (label) для прогнозирования будущего поведения
# Здесь в демонстрационных целях создается искусственная метка, основанная на порогах ключевых признаков.
# В реальном случае метка может определяться историческими данными (напр., отток, перекрестные продажи, апселл и др.)
median_avg_amount = agg_df['avg_amount'].median()
median_tx_count = agg_df['transaction_count'].median()

def determine_future_action(row):
    if row['transaction_count'] < median_tx_count:
        return 'Churn'         # Клиент малое число операций – риск ухода
    elif row['avg_amount'] > median_avg_amount:
        return 'Upsell'        # Высокая средняя сумма может говорить о потенциале к увеличению лимита/расходов
    else:
        return 'Loyal'         # Стабильные показатели – клиент, которого стоит удерживать

agg_df['future_action'] = agg_df.apply(determine_future_action, axis=1)

# 6. Подготовка данных для построения модели
features = [
    'total_amount', 'avg_amount', 'transaction_count', 'std_amount',
    'unique_transaction_types', 'unique_merchant_mcc', 'unique_merchant_city',
    'avg_tx_gap_days', 'domestic_ratio', 'unique_pos_entry_mode', 'unique_wallet_type'
]
X = agg_df[features]
y = agg_df['future_action']

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Кодирование целевой переменной (не обязательно для RandomForest, но удобно для анализа)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Разбивка на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# 7. Обучение модели для прогнозирования будущих действий клиента
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Оценка модели
y_pred = rf.predict(X_test)
print("Отчет по классификации:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("Матрица ошибок:\n", confusion_matrix(y_test, y_pred))

# 8. Применение модели для прогнозирования для всех клиентов
agg_df['predicted_future_action'] = le.inverse_transform(rf.predict(X_scaled))

# 9. Генерация рекомендаций для банков
# На основе прогноза будущих действий клиента выдаются рекомендации, как удовлетворить его потребности.
bank_recommendations = {
    'Churn': "Провести анализ причин оттока и предложить специальные скидки или бонусы для удержания клиента.",
    'Upsell': "Рассмотреть возможность увеличения кредитного лимита и предложение премиальных продуктов.",
    'Loyal': "Поддерживать клиента через программы лояльности, дополнительные бонусы и персональные предложения."
}
agg_df['bank_recommendation'] = agg_df['predicted_future_action'].map(bank_recommendations)

# 10. Вывод результатов
result = agg_df[['card_id', 'predicted_future_action', 'bank_recommendation']]
print("Результаты прогнозирования и рекомендации:")
print(result.head(20))

# Визуализация распределения прогнозов (пример)
plt.figure(figsize=(8, 5))
plt.figure(figsize=(8, 5))
ax = sns.countplot(
    x='predicted_future_action',
    data=agg_df,
    hue='predicted_future_action',  # явно задаём раскраску по значению переменной
    palette='Set2'
)
legend = ax.get_legend()  # получаем легенду, если она существует
if legend is not None:
    legend.remove()       # удаляем её, так как информация уже присутствует по оси X
plt.title("Распределение прогнозируемых будущих действий клиентов")
plt.xlabel("Будущее поведение")
plt.ylabel("Количество клиентов")
plt.show()

plt.title("Распределение прогнозируемых будущих действий клиентов")
plt.xlabel("Будущее поведение")
plt.ylabel("Количество клиентов")
plt.show()

# Сохранение результатов
agg_df.to_parquet("data/processed/clients_future_action_recommendations.parquet", index=False)
print("Результаты сохранены в 'data/processed/clients_future_action_recommendations.parquet'")




