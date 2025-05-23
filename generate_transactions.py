import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Установим сид для воспроизводимости
np.random.seed(42)
random.seed(42)
fake = Faker()

# Настройки
n_clients = 500
transactions_per_client = np.random.randint(10, 30, size=n_clients)

# Категории
mcc_codes = [5411, 5812, 5999, 4111, 6011, 5732, 4899]
transaction_types = ['PURCHASE', 'ATM_WITHDRAWAL', 'P2P', 'SALARY']
wallet_types = ['ApplePay', 'GooglePay', 'SamsungPay', None]
pos_modes = ['Chip', 'Magstripe', 'Contactless', 'Manual']
currencies = ['KZT', 'USD', 'EUR']

rows = []
start_date = datetime(2023, 1, 1)

# Генерация транзакций
for client_id, txn_count in zip(range(1000, 1000 + n_clients), transactions_per_client):
    for _ in range(txn_count):
        txn_time = start_date + timedelta(
            days=np.random.randint(0, 180),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        amount = round(np.random.exponential(5000) + 500, 2)
        currency = np.random.choice(currencies, p=[0.85, 0.1, 0.05])
        mcc = np.random.choice(mcc_codes)
        wallet = np.random.choice(wallet_types, p=[0.3, 0.2, 0.1, 0.4])
        pos = np.random.choice(pos_modes)
        t_type = np.random.choice(transaction_types, p=[0.75, 0.15, 0.08, 0.02])
        merchant_city = fake.city() if np.random.rand() > 0.1 else None

        rows.append({
            'card_id': client_id,
            'transaction_id': fake.uuid4(),
            'transaction_timestamp': txn_time,
            'transaction_amount_kzt': amount if currency == 'KZT' else None,
            'original_amount': amount,
            'transaction_currency': currency,
            'merchant_mcc': mcc,
            'transaction_type': t_type,
            'wallet_type': wallet,
            'pos_entry_mode': pos,
            'merchant_city': merchant_city
        })

# Сохраняем как DataFrame
df = pd.DataFrame(rows)

# Сохраняем в parquet (pyarrow должен быть установлен!)
df.to_parquet("data/raw/transactions.parquet", index=False)

print("✅ Датасет сгенерирован и сохранён: data/raw/transactions.parquet")
