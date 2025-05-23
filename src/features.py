import pandas as pd

def compute_features(df):
    df = df.sort_values(['card_id', 'transaction_timestamp'])
    df['days_between'] = df.groupby('card_id')['transaction_timestamp'].diff().dt.days

    grouped = df.groupby('card_id')

    features = pd.DataFrame()
    features['total_txn_count'] = grouped['transaction_id'].count()
    features['avg_amount'] = grouped['transaction_amount_kzt'].mean()
    features['std_amount'] = grouped['transaction_amount_kzt'].std()
    features['total_amount'] = grouped['transaction_amount_kzt'].sum()
    features['unique_mcc'] = grouped['merchant_mcc'].nunique()
    features['unique_city'] = grouped['merchant_city'].nunique()
    features['avg_days_between'] = df.groupby('card_id')['days_between'].mean()

    features['pct_wallet'] = grouped['wallet_type'].apply(lambda x: x.notna().sum() / len(x))
    features['pct_contactless'] = grouped['pos_entry_mode'].apply(lambda x: (x == 'Contactless').sum() / len(x))
    features['pct_cash'] = grouped['transaction_type'].apply(lambda x: (x == 'ATM_WITHDRAWAL').sum() / len(x))
    features['pct_foreign'] = grouped['transaction_currency'].apply(lambda x: (x != 'KZT').sum() / len(x))

    return features.fillna(0).reset_index()
