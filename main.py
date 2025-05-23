import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
# Для воспроизводимости
random.seed(42)
np.random.seed(42)
# Импорт библиотек HuggingFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch


# Функция для вычисления среднего интервала между транзакциями
def compute_avg_days_between_transactions(df):
    """
    Вычисляет среднее число дней между транзакциями по каждому card_id.
    """
    df_sorted = df.sort_values(by=['card_id', 'transaction_timestamp'])
    df_sorted['days_diff'] = df_sorted.groupby('card_id')['transaction_timestamp'].diff().dt.days
    avg_days = df_sorted.groupby('card_id')['days_diff'].mean().reset_index().rename(
        columns={'days_diff': 'avg_days_between_txn'}
    )
    return avg_days


# Функция для вывода архитектурной диаграммы проекта (ASCII-диаграмма)
def generate_architecture_diagram():
    """
    Выводит ASCII-диаграмму архитектуры проекта.
    """
    diagram = """
    +------------------------------------------------------------+
    |                        Data Sources                        |
    |                - data/raw/transactions.parquet             |
    +---------------------------+--------------------------------+
                                |
                                v
    +---------------------------+--------------------------------+
    |                   Data Ingestion                         |
    |         (pandas, numpy, datasets library)                |
    +---------------------------+--------------------------------+
                                |
                                v
    +---------------------------+--------------------------------+
    |           Data Cleaning & Preprocessing                  |
    |   - Приведение типов, удаление пропусков, фильтрация       |
    +---------------------------+--------------------------------+
                                |
                                v
    +---------------------------+--------------------------------+
    |        Feature Engineering & Aggregation                 |
    |  - Агрегация транзакций по card_id:                        |
    |    * total_transactions, total_amount, avg, std            |
    |    * num_unique_mcc, num_unique_cities                      |
    |    * pct_wallet_use, pct_cash_withdrawals, avg_days_between  |
    +---------------------------+--------------------------------+
                                |
                                v
    +---------------------------+--------------------------------+
    |         Data Scaling (StandardScaler)                    |
    +---------------------------+--------------------------------+
                                |
                                v
    +---------------------------+--------------------------------+
    |        Segmentation Model (Unsupervised Learning)        |
    |  - KMeans (основной) / DBSCAN (альтернативный)             |
    |  - Оценка качества (Silhouette Score)                      |
    +---------------------------+--------------------------------+
                                |
                                v
    +---------------------------+--------------------------------+
    |       Visualization & Interpretation                     |
    |   - PCA для 2D проекции, отображение кластеров             |
    |   - Бизнес-инсайты по сегментам                            |
    +---------------------------+--------------------------------+
                                |
                                v
    +---------------------------+--------------------------------+
    |  Integration with HuggingFace Transformers/Datasets      |
    |   - Demo: BERT для анализа текста                          |
    |   - Demo: Загрузка набора данных через Datasets              |
    +---------------------------+--------------------------------+
                                |
                                v
    +---------------------------+--------------------------------+
    |         Output & Presentation (Notebook, PDF)            |
    |   - Итоговый файл с сегментацией (Parquet)                 |
    |   - Презентация результатов с графиками                   |
    +------------------------------------------------------------+
    """
    print(diagram)


def main():
    # ----------------------
    # PART 1: Data Ingestion and Cleaning
    # ----------------------
    print("Загрузка данных из data/raw/transactions.parquet ...")
    data_path = 'data/raw/transactions.parquet'

    try:
        # Попытка загрузить данные из файла Parquet
        df = pd.read_parquet(data_path)
        print(f"Данные успешно загружены из {data_path}")
    except Exception as e:
        print(f"Ошибка при загрузке данных из {data_path}: {e}")
        return  # Прекращаем выполнение, если загрузка не удалась

    # Приведение столбца transaction_timestamp к типу datetime
    df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'], errors='coerce')
    # Приведение transaction_amount_kzt к числовому типу
    df['transaction_amount_kzt'] = pd.to_numeric(df['transaction_amount_kzt'], errors='coerce')
    # Фильтрация транзакций с суммой > 0
    df = df[df['transaction_amount_kzt'] > 0]

    # ----------------------
    # PART 2: Feature Engineering
    # ----------------------
    print("Выполнение feature engineering...")

    # Агрегированные метрики по card_id:
    aggregation = {
        'transaction_id': 'count',  # общее число транзакций
        'transaction_amount_kzt': ['sum', 'mean', 'std'],
        'merchant_mcc': pd.Series.nunique,
        'merchant_city': pd.Series.nunique
    }

    aggregated = df.groupby('card_id').agg(aggregation)
    # Приводим мультииндекс столбцов к обычным именам
    aggregated.columns = ['total_transactions', 'total_amount_kzt', 'avg_amount_kzt', 'std_amount_kzt',
                          'num_unique_mcc', 'num_unique_cities']
    aggregated = aggregated.reset_index()

    # Процентное использование цифрового кошелька: доля транзакций, где wallet_type не пустой
    wallet_usage = df.groupby('card_id').apply(lambda x: (x['wallet_type'] != '').sum() / len(x))
    wallet_usage = wallet_usage.reset_index(name='pct_wallet_use')

    # Процент транзакций - снятие наличных (ATM_WITHDRAWAL)
    cash_withdrawals = df.groupby('card_id').apply(lambda x: (x['transaction_type'] == 'ATM_WITHDRAWAL').sum() / len(x))
    cash_withdrawals = cash_withdrawals.reset_index(name='pct_cash_withdrawals')

    # Расчет среднего интервала между транзакциями
    avg_days = compute_avg_days_between_transactions(df)

    # Объединяем все вычисленные признаки по card_id
    features_df = aggregated.merge(wallet_usage, on='card_id', how='left') \
        .merge(cash_withdrawals, on='card_id', how='left') \
        .merge(avg_days, on='card_id', how='left')

    # Обработка возможных NaN — например, для std_amount_kzt или avg_days_between_txn
    features_df['std_amount_kzt'] = features_df['std_amount_kzt'].fillna(0)
    features_df['avg_days_between_txn'] = features_df['avg_days_between_txn'].fillna(0)

    print("Сформированные признаки:")
    print(features_df.head())

    # ----------------------
    # PART 3: Data Scaling and Segmentation (Clustering)
    # ----------------------
    print("Масштабирование данных и сегментация с использованием KMeans...")

    # Выбор признаков для кластеризации
    feature_columns = ['total_transactions', 'total_amount_kzt', 'avg_amount_kzt', 'std_amount_kzt',
                       'num_unique_mcc', 'num_unique_cities', 'pct_wallet_use', 'pct_cash_withdrawals',
                       'avg_days_between_txn']

    X = features_df[feature_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Сегментация с помощью KMeans
    n_clusters = 4  # Можно оптимизировать выбор числа кластеров
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    features_df['cluster_kmeans'] = kmeans.fit_predict(X_scaled)

    silhouette_kmeans = silhouette_score(X_scaled, features_df['cluster_kmeans'])
    print(f"Silhouette score для KMeans: {silhouette_kmeans:.2f}")

    # Альтернативная сегментация с использованием DBSCAN
    print("Альтернативная сегментация с использованием DBSCAN...")
    dbscan = DBSCAN(eps=0.8, min_samples=3)
    features_df['cluster_dbscan'] = dbscan.fit_predict(X_scaled)

    if len(set(features_df['cluster_dbscan'])) > 1 and -1 not in set(features_df['cluster_dbscan']):
        silhouette_dbscan = silhouette_score(X_scaled, features_df['cluster_dbscan'])
        print(f"Silhouette score для DBSCAN: {silhouette_dbscan:.2f}")
    else:
        print("DBSCAN: недостаточно кластеров для расчёта silhouette score или присутствуют шумовые точки.")

    # ----------------------
    # PART 4: Visualization using PCA
    # ----------------------
    print("Визуализация кластеров с использованием PCA...")
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_scaled)
    features_df['pca_one'] = pca_components[:, 0]
    features_df['pca_two'] = pca_components[:, 1]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(data=features_df, x='pca_one', y='pca_two', hue='cluster_kmeans', palette='viridis', s=100)
    plt.title("Сегментация клиентов (KMeans)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    plt.subplot(1, 2, 2)
    sns.scatterplot(data=features_df, x='pca_one', y='pca_two', hue='cluster_dbscan', palette='coolwarm', s=100)
    plt.title("Сегментация клиентов (DBSCAN)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    plt.tight_layout()
    plt.show()

    # ----------------------
    # PART 5: HuggingFace Transformers Demo
    # ----------------------
    print("Демонстрация работы с HuggingFace Transformers (BERT)...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    sample_text = "This transaction was processed using a digital wallet."
    inputs = tokenizer(sample_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    print("Логиты модели BERT для примера текста:")
    print(outputs.logits)

    # ----------------------
    # PART 6: HuggingFace Datasets Demo
    # ----------------------
    print("Загрузка набора данных через HuggingFace Datasets (IMDB)...")
    imdb_dataset = load_dataset("imdb", split="train[:1%]")
    print("Пример записи из набора данных IMDB:")
    print(imdb_dataset[0])

    # ----------------------
    # PART 7: Save Results and Report
    # ----------------------
    results_file = "segmentation_results.parquet"
    features_df.to_parquet(results_file, index=False)
    print(f"Результаты сегментации сохранены в файл {results_file}")


if __name__ == "__main__":
    print("Архитектура решения:")
    generate_architecture_diagram()
    main()
