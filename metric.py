import pandas as pd
import os

# Папка, где лежат все 1999 .csv
csv_folder = "output_parquet/csv"

# Список метрик по всем пользователям
all_metrics = []

for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        path = os.path.join(csv_folder, file)
        df = pd.read_csv(path)

        if df.empty or len(df) < 2:
            continue

        df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"])
        df = df.sort_values("transaction_timestamp")
        df["days_between"] = df["transaction_timestamp"].diff().dt.days

        card_id = df["card_id"].iloc[0]

        metrics = {
            "card_id": card_id,
            "total_txn_count": len(df),
            "total_amount": df["transaction_amount_kzt"].sum(),
            "avg_amount": df["transaction_amount_kzt"].mean(),
            "std_amount": df["transaction_amount_kzt"].std(),
            "unique_mcc": df["merchant_mcc"].nunique(),
            "unique_city": df["merchant_city"].nunique(),
            "avg_days_between": df["days_between"].mean(),
            "pct_wallet": df["wallet_type"].notna().mean(),
            "pct_foreign": (df["transaction_currency"] != "KZT").mean(),
            "pct_cash": (df["transaction_type"] == "ATM_WITHDRAWAL").mean(),
            "pct_p2p": df["transaction_type"].isin(["P2P_IN", "P2P_OUT"]).mean()
        }

        all_metrics.append(metrics)

# Собираем всё в итоговый датафрейм
metrics_df = pd.DataFrame(all_metrics)

# Сохраняем
metrics_df.to_csv("customer_metrics.csv", index=False)
print("✅ customer_metrics.csv — готово!")
