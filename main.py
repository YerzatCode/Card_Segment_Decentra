import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загрузка данных
df1 = pd.read_csv('data/processed/client_features.csv')
df2 = pd.read_csv('data/processed/transactions_processed.csv')
df = pd.concat([df1, df2], ignore_index=True)

# 2. Предобработка
df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'], errors='coerce')

# 3. Агрегация по клиенту (card_id)
agg_df = df.groupby('card_id').agg({
    'transaction_amount_kzt': ['sum', 'mean', 'count'],
    'transaction_type': pd.Series.nunique,
    'merchant_mcc': pd.Series.nunique
}).reset_index()

# 4. Переименование колонок
agg_df.columns = ['card_id', 'total_amount', 'avg_amount', 'transaction_count',
                  'unique_types', 'unique_mcc']

# 5. Обработка NaN (важно!)
features = ['total_amount', 'avg_amount', 'transaction_count', 'unique_types', 'unique_mcc']

# Проверим, есть ли NaN
print("NaN до обработки:\n", agg_df[features].isnull().sum())

# Заполним средними значениями
agg_df[features] = agg_df[features].fillna(agg_df[features].mean())

# Если NaN всё ещё есть — удалим такие строки
agg_df.dropna(subset=features, inplace=True)

# Проверим ещё раз
print("NaN после обработки:\n", agg_df[features].isnull().sum())

# 6. Масштабирование признаков
X = agg_df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Кластеризация
kmeans = KMeans(n_clusters=5, random_state=42)
agg_df['cluster'] = kmeans.fit_predict(X_scaled)

# 8. Интерпретация кластеров
cluster_profiles = agg_df.groupby('cluster')[features].mean().reset_index()

# Назначим ярлыки вручную (пример — можно отрегулировать)
segment_labels = {
    0: 'редкие покупатели с маленькими суммами',
    1: 'частые покупки, но на малые суммы',
    2: 'частые покупки, но на крупные суммы',
    3: 'не частые покупки, но на крупные суммы',
    4: 'корпоративные или VIP-клиенты с высокими суммами'
}

agg_df['segment'] = agg_df['cluster'].map(segment_labels)

# 9. Визуализация кластеров
sns.pairplot(agg_df, hue='segment', vars=features, palette='tab10')
plt.suptitle("Сегментация клиентов по поведению", y=1.02)
plt.show()

# 10. Вывод результата
result = agg_df[['card_id', 'segment']]
print(result)

# Опционально — сохранить
# result.to_csv("client_segments.csv", index=False)

# 10. Сохранение результата в .parquet
agg_df.to_parquet("data/processed/client_segments.parquet", index=False)
print("Сегменты клиентов успешно сохранены в 'data/processed/client_segments.parquet'")



