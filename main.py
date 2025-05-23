def run_pipeline():
    from src.config import RAW_DATA, FEATURES_PATH, CLUSTERED_PATH, N_CLUSTERS
    from src.utils import load_parquet, save_parquet
    from src.preprocessing import preprocess
    from src.features import compute_features
    from src.clustering import cluster_customers
    from src.interpretation import assign_cluster_names

    print('🔹 Загрузка данных')
    df = load_parquet(RAW_DATA)

    print('🔹 Предобработка')
    df_clean = preprocess(df)

    print('🔹 Feature Engineering')
    features = compute_features(df_clean)
    save_parquet(features, FEATURES_PATH)

    print('🔹 Кластеризация')
    clustered_df, model = cluster_customers(features, N_CLUSTERS)

    print('🔹 Назначение имён кластерам')
    clustered_df = assign_cluster_names(clustered_df)
    save_parquet(clustered_df, CLUSTERED_PATH)

    print('✅ Кластеризация завершена!')


if __name__ == "__main__":
    run_pipeline()
