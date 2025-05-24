import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import dask.dataframe as dd

# 1. Загрузка данных из разных источников
df1 = dd.read_parquet('data/processed/client_features.parquet')
df2 = dd.read_parquet('data/processed/transactions_processed.parquet')
df = dd.concat([df1, df2], ignore_index=True).compute()

# 2. Приведение дат к корректному формату
df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'], errors='coerce')
# Для expiry_date (формат MM/YY) можно задать первое число месяца для корректного преобразования
df['expiry_date'] = pd.to_datetime('20' + df['expiry_date'], format='%Y%m', errors='coerce')

# 3. Агрегация транзакционных данных по card_id с учётом разнообразных полей
# Функция для вычисления среднего интервала между транзакциями (в днях)
def compute_avg_gap(x):
    if len(x) > 1:
        x = x.sort_values()
        gaps = x.diff().dropna()
        return gaps.mean().days
    else:
        return np.nan

agg_funcs = {
    # Работа с суммами – сумма, среднее, количество и разброс сумм
    'transaction_amount_kzt': ['sum', 'mean', 'count', 'std'],
    # Уникальные типы операций (POS, ATM_WITHDRAWAL, P2P и т.д.)
    'transaction_type': pd.Series.nunique,
    # Разнообразие торговых категорий
    'merchant_mcc': pd.Series.nunique,
    # Количество уникальных городов, где были проведены операции
    'merchant_city': pd.Series.nunique,
    # Средний интервал между операциями (в днях)
    'transaction_timestamp': compute_avg_gap,
    # Доля транзакций, проведённых внутри страны (например, где ISO-код страны равен 'KAZ')
    'acquirer_country_iso': lambda x: (x == 'KAZ').sum() / x.count() if x.count() > 0 else 0,
    # Разнообразие режимов ввода (Chip, Magstripe, Contactless и т.д.)
    'pos_entry_mode': pd.Series.nunique,
    # Разнообразие типов кошельков (ApplePay, GooglePay, и т.д.)
    'wallet_type': pd.Series.nunique
}

# Группировка по card_id и применение агрегирующих функций
agg_df = df.groupby('card_id').agg(agg_funcs)
agg_df.columns = [
    'total_amount', 'avg_amount', 'transaction_count', 'std_amount',
    'unique_transaction_types', 'unique_merchant_mcc', 'unique_merchant_city',
    'avg_tx_gap_days', 'domestic_ratio', 'unique_pos_entry_mode', 'unique_wallet_type'
]
agg_df.reset_index(inplace=True)

# 4. Обработка пропущенных значений
# Если каких-то агрегатов не получилось посчитать (например, средний интервал при единственной транзакции), заполняем средним
agg_df.fillna(agg_df.mean(), inplace=True)

# 5. Выбор признаков для кластеризации
features = ['total_amount', 'avg_amount', 'transaction_count', 'std_amount',
            'unique_transaction_types', 'unique_merchant_mcc', 'unique_merchant_city',
            'avg_tx_gap_days', 'domestic_ratio', 'unique_pos_entry_mode', 'unique_wallet_type']

X = agg_df[features]

# 6. Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Кластеризация
# Здесь выбрано 7 кластеров для разделения клиентов на более дифференцированные группы
kmeans = KMeans(n_clusters=7, random_state=42)
agg_df['cluster'] = kmeans.fit_predict(X_scaled)

# Отобразим сводную статистику по кластерам для первичной интерпретации
cluster_profiles = agg_df.groupby('cluster')[features].mean().reset_index()
print("Профили кластеров:")
print(cluster_profiles)

# 8. Присвоение поведенческих метрик (наименований)
# Наименования условны и основаны на оценке профилей: их можно адаптировать на основе подробного анализа.
metric_labels = {
    0: 'Зожник',              # Возможен вариант: умеренные траты, регулярные операции, стабильность
    1: 'Путешественник',       # Высокое число уникальных городов и стабильная активность
    2: 'Тусовщик',             # Частые, но небольшие операции, широкий спектр операционных типов
    3: 'Люкс',                 # Высокие суммарные и средние затраты, возможна высокая волатильность
    4: 'Экономный',            # Низкие суммы и относительно малое количество операций
    5: 'Техногик',             # Много уникальных способов оплаты: цифровые транзакции через разные wallet-ы и POS
    6: 'Любитель приключений'  # Значительные промежутки между операциями, редкие, но крупные покупки
}

agg_df['behavior_metric'] = agg_df['cluster'].map(metric_labels)

# 9. Визуализация для проверки распределения по некоторому из признаков
plt.figure(figsize=(10, 6))
ax = sns.boxplot(
    x='behavior_metric',
    y='total_amount',
    data=agg_df,
    hue='behavior_metric',  # задаём раскраску для каждой категории
    palette='Set3',
    dodge=False  # чтобы боксы не делились друг на друга внутри категории
)
legend = ax.get_legend()
if legend is not None:
    legend.remove()
plt.title("Распределение суммарных трат по типам клиентов")
plt.xticks(rotation=45)
plt.show()


# Также можно воспользоваться pairplot для просмотра взаимосвязей между признаками
sns.pairplot(agg_df, hue='behavior_metric', vars=features, palette='tab10')
plt.suptitle("Сгруппированные по типам клиентов (поведение и транзакции)", y=1.02)
plt.show()

# 10. Вывод результата: для каждого card_id указывается присвоенная поведенческая метрика
result = agg_df[['card_id', 'behavior_metric']]
print("Результат кластеризации по типам поведения:")
print(result.head(20))

# Опциональное сохранение результата
agg_df.to_parquet("data/processed/clients_behavior_metrics.parquet", index=False)
print("Поведенческие метрики клиентов сохранены в 'data/processed/clients_behavior_metrics.parquet'")
