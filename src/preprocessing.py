import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['transaction_amount_kzt'] > 0]
    df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
    return df
