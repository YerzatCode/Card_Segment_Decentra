import pandas as pd

def load_parquet(path):
    return pd.read_parquet(path)

def save_parquet(df, path):
    df.to_parquet(path, index=False)
