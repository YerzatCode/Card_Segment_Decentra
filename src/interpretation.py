def assign_cluster_names(df):
    labels = {
        0: 'Цифровые активисты',
        1: 'Экономные потребители',
        2: 'Бюджетные традиционалисты',
        3: 'Премиум-покупатели',
        4: 'Умеренные и стабильные'
    }
    df['cluster_name'] = df['cluster'].map(labels)
    return df
