from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def cluster_customers(df: pd.DataFrame, n_clusters=5):
    X = df.drop(columns=['card_id'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = model.fit_predict(X_scaled)

    return df, model
