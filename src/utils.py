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
df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])

# Агрегируем данные по клиенту (card_id)
agg_df = df.groupby('card_id').agg({
    'transaction_amount_kzt': ['sum', 'mean', 'count'],
    'transaction_type': pd.Series.nunique,
    'merchant_mcc': pd.Series.nunique
}).reset_index()

agg_df.columns = ['card_id', 'total_amount', 'avg_amount', 'transaction_count',
                  'unique_types', 'unique_mcc']

# 3. Масштабирование
features = ['total_amount', 'avg_amount', 'transaction_count', 'unique_types', 'unique_mcc']
X = agg_df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Построение модели K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
agg_df['cluster'] = kmeans.fit_predict(X_scaled)

# 5. Визуализация
sns.pairplot(agg_df, hue='cluster', vars=features, palette='tab10')
plt.show()