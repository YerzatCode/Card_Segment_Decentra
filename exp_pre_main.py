import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
# --- 1. Сбор и подготовка данных ---
# 🔹 Входные данные: Таблица с транзакциями

# Укажите правильный путь к вашему файлу
file_path = 'data/raw/transactions.parquet' # Замените на ваш путь, если необходимо

# --- Шаги обработки: ---

# --- Загрузка данных (pandas) ---
try:
    df_transactions = dd.read_parquet(file_path).compute()
    print("Данные успешно загружены.")
    print(f"Исходный размер данных: {df_transactions.shape[0]} строк, {df_transactions.shape[1]} столбцов.")
except FileNotFoundError:
    print(f"Файл не найден по пути: {file_path}")
    print("Пожалуйста, проверьте правильность пути к файлу 'transactions.parquet'.")
    exit()
except Exception as e:
    print(f"Произошла ошибка при загрузке файла: {e}")
    exit()

print("\n--- Первые 5 строк исходных данных: ---")
print(df_transactions.head())

print("\n--- Информация о типах данных и пропущенных значениях: ---")
df_transactions.info()

# --- Очистка от пустых/аномальных значений (начальный этап) и Приведение типов данных ---

# Колонки, описанные в документации [cite: 33, 34]
# transaction_id: Уникальный идентификатор транзакции (UUID) - обычно object/string
# transaction_timestamp: Дата и время проведения операции (YYYY-MM-DD HH:MM:SS) - datetime
# card_id: Внутренний числовой идентификатор карты - int or object (если есть нечисловые префиксы/суффиксы)
# expiry_date: Срок действия карты в формате MM/YY - object/string, можно преобразовать в datetime
# issuer_bank_name: Название банка-эмитента карты - object/string
# merchant_id: Идентификатор торговой точки - object/string (может быть пустым)
# merchant_mcc: Код MCC (Merchant Category Code) - int or object (лучше category)
# merchant_city: Город торговой точки - object/string (может быть пустым)
# transaction_type: Тип операции - object/string (лучше category)
# transaction_amount_kzt: Сумма транзакции в тенге (KZT) - float/int
# original_amount: Исходная сумма операции в валюте платежа - float (может быть пустым)
# transaction_currency: Код валюты платежа (ISO-Alpha-3) - object/string (лучше category)
# acquirer_country_iso: Код страны-эквайера (ISO-Alpha-3) - object/string (лучше category)
# pos_entry_mode: Режим ввода данных карты на POS-терминале - object/string (лучше category)
# wallet_type: Тип цифрового кошелька - object/string (может быть пустым, лучше category)

# Приведение типов данных
print("\n--- Приведение типов данных: ---")

# Дата и время
if 'transaction_timestamp' in df_transactions.columns:
    df_transactions['transaction_timestamp'] = pd.to_datetime(df_transactions['transaction_timestamp'])
    print("'transaction_timestamp' преобразован в datetime.")

# Числовые типы для сумм
if 'transaction_amount_kzt' in df_transactions.columns:
    df_transactions['transaction_amount_kzt'] = pd.to_numeric(df_transactions['transaction_amount_kzt'], errors='coerce')
    print("'transaction_amount_kzt' преобразован в числовой тип.")
if 'original_amount' in df_transactions.columns:
    df_transactions['original_amount'] = pd.to_numeric(df_transactions['original_amount'], errors='coerce')
    print("'original_amount' преобразован в числовой тип.")

# Специальное приведение для 'merchant_mcc':
# Если ожидается, что колонка содержит как числовые данные, так и значения типа "Unknown",
# удобнее сохранить её как строку (str), чтобы избежать ошибок преобразования.
if 'merchant_mcc' in df_transactions.columns:
    df_transactions['merchant_mcc'] = df_transactions['merchant_mcc'].astype(str)
    print("'merchant_mcc' преобразован в строковый тип (str).")

# Категориальные типы (примеры)
categorical_cols = ['transaction_type', 'transaction_currency',
                    'acquirer_country_iso', 'pos_entry_mode', 'wallet_type', 'issuer_bank_name', 'merchant_city']
for col in categorical_cols:
    if col in df_transactions.columns:
        df_transactions[col] = df_transactions[col].astype('category')
        print(f"'{col}' преобразован в категориальный тип.")

# Обработка card_id - предполагаем, что это числовой идентификатор, но может быть считан как object
if 'card_id' in df_transactions.columns:
     # Если card_id может содержать нечисловые значения или очень большие числа, лучше оставить object или string
     # Если это точно числовой ID, можно преобразовать:
     # df_transactions['card_id'] = pd.to_numeric(df_transactions['card_id'], errors='raise') # или 'coerce'
    print(f"Тип 'card_id' оставлен как {df_transactions['card_id'].dtype} (проверьте на соответствие вашим данным).")


# --- Удаление транзакций с нулевыми или отрицательными суммами --- [cite: 2]
if 'transaction_amount_kzt' in df_transactions.columns:
    initial_rows = df_transactions.shape[0]
    df_transactions = df_transactions[df_transactions['transaction_amount_kzt'] > 0]
    rows_after_cleaning = df_transactions.shape[0]
    print(f"\nУдалено {initial_rows - rows_after_cleaning} транзакций с нулевой или отрицательной суммой.")
    print(f"Размер данных после удаления некорректных сумм: {rows_after_cleaning} строк.")
else:
    print("\nКолонка 'transaction_amount_kzt' не найдена, пропуск удаления транзакций с некорректными суммами.")

# --- Обработка пропущенных значений (пример стратегии) ---
# В документации указано, что некоторые поля могут быть пустыми (merchant_id, merchant_city, original_amount, wallet_type) [cite: 33, 34]
print("\n--- Обработка пропущенных значений: ---")
for column in df_transactions.columns:
    if df_transactions[column].isnull().any():
        null_count = df_transactions[column].isnull().sum()
        print(f"Колонка '{column}' содержит {null_count} пропущенных значений ({null_count / len(df_transactions):.2%}).")
        # Пример: для категориальных можно заполнить модой или специальным значением 'Unknown'
        if df_transactions[column].dtype.name == 'category':
            # Заполняем пропуски в категориальных новым значением 'Unknown'
            # Сначала нужно добавить 'Unknown' в категории, если его там нет
            if 'Unknown' not in df_transactions[column].cat.categories:
                 df_transactions[column] = df_transactions[column].cat.add_categories('Unknown')
            df_transactions[column] = df_transactions[column].fillna('Unknown')
            print(f"Пропуски в '{column}' заменены на 'Unknown'.")
        # Пример: для числовых можно заполнить медианой или нулем, в зависимости от смысла поля
        elif pd.api.types.is_numeric_dtype(df_transactions[column]):
            # Для 'original_amount' пропуск может означать, что транзакция была в KZT, заполним 0 или оставим как есть
            if column == 'original_amount':
                 df_transactions[column] = df_transactions[column].fillna(0) # или другое значение/стратегия
                 print(f"Пропуски в '{column}' (числовой) заменены на 0.")
            # Для других числовых колонок нужна своя стратегия.
            # Например, если бы у нас были числовые признаки для кластеризации с пропусками, их нужно было бы обработать (медиана, среднее и т.д.)

# Проверим пропуски после обработки
print("\n--- Проверка пропусков после обработки: ---")
print(df_transactions.isnull().sum())

print("\n--- Первые 5 строк обработанных данных: ---")
print(df_transactions.head())

print("\n--- Информация о типах данных после обработки: ---")
df_transactions.info()

print("\n--- Описательная статистика для числовых колонок после обработки: ---")
# Добавим проверку на наличие числовых колонок перед вызовом describe
numeric_cols_processed = df_transactions.select_dtypes(include=np.number).columns
if not numeric_cols_processed.empty:
    print(df_transactions[numeric_cols_processed].describe())
else:
    print("Числовые колонки для описательной статистики не найдены.")


# Следующий шаг по архитектуре - "Агрегация транзакций по card_id — получение одного вектора признаков на клиента" [cite: 2]
# Этот шаг является началом Feature Engineering и будет выполнен далее.
# На данном этапе данные загружены, очищены и приведены к нужным типам.

print("\n--- Обработка данных завершена. Данные готовы для Feature Engineering. ---")

# Сохранение обработанного DataFrame (опционально)
# processed_file_path = 'data/processed/transactions_processed.parquet'
# df_transactions.to_parquet(processed_file_path)
# print(f"\nОбработанные данные сохранены в: {processed_file_path}")

print("\n--- Этап 2: Feature Engineering ---")

# Агрегация транзакций по card_id для создания клиентского вектора признаков
# Задаем словарь с функциями агрегации по нескольким полям
agg_functions = {
    'transaction_id': 'count',  # количество транзакций
    'transaction_amount_kzt': ['sum', 'mean', 'median', 'std', 'min', 'max'],  # статистика по суммам
    'merchant_id': pd.Series.nunique,  # число уникальных торговых точек
    'merchant_mcc': pd.Series.nunique,   # число уникальных категорий MCC
    'transaction_timestamp': ['min', 'max']  # первая и последняя транзакция по времени
}

df_features = df_transactions.groupby('card_id').agg(agg_functions)

# Колонки после множественной агрегации имеют многоуровневые названия, упрощаем их:
df_features.columns = ['_'.join(col).strip() for col in df_features.columns.values]

# Добавляем новые признаки:
# 1. Количество активных дней (от разницы между первой и последней транзакцией)
df_features['active_days'] = (df_features['transaction_timestamp_max'] - df_features['transaction_timestamp_min']).dt.days + 1
# 2. Среднее число транзакций в день
df_features['transactions_per_day'] = df_features['transaction_id_count'] / df_features['active_days']

# Для удобства переименовываем столбцы в более осмысленные названия:
df_features.rename(columns={
    'transaction_id_count': 'total_transactions',
    'transaction_amount_kzt_sum': 'total_amount_kzt',
    'transaction_amount_kzt_mean': 'mean_amount_kzt',
    'transaction_amount_kzt_median': 'median_amount_kzt',
    'transaction_amount_kzt_std': 'std_amount_kzt',
    'transaction_amount_kzt_min': 'min_amount_kzt',
    'transaction_amount_kzt_max': 'max_amount_kzt',
    'merchant_id_nunique': 'unique_merchants',
    'merchant_mcc_nunique': 'unique_mcc',
    'transaction_timestamp_min': 'first_transaction',
    'transaction_timestamp_max': 'last_transaction'
}, inplace=True)

print("\n--- Первые 5 строк агрегированных данных (на клиентском уровне): ---")
print(df_features.head())

# Дополнительно можно исследовать динамику транзакционной активности, например:
# - Расчет интервалов между транзакциями
# - Выявление сезонных или дневных трендов
#
# Если потребуется, можно добавить дополнительные признаки, которые могут быть полезны для модели,
# например, признаки, связанные с категоризацией торговых точек (используя данные из merchant_mcc).

# Опционально: сохранение агрегированных признаков для дальнейшего использования
# processed_features_path = 'data/processed/client_features.parquet'
# df_features.to_parquet(processed_features_path)
# print(f"\nАгрегированные признаки сохранены в: {processed_features_path}")

# Построим гистограмму распределения количества транзакций по клиентам
plt.figure(figsize=(10, 6))
plt.hist(df_features['total_transactions'], bins=50, color='skyblue', edgecolor='black')
plt.title('Распределение количества транзакций по клиентам')
plt.xlabel('Количество транзакций')
plt.ylabel('Частота (количество клиентов)')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Построим гистограмму распределения общей суммы транзакций (KZT)
plt.figure(figsize=(10, 6))
plt.hist(df_features['total_amount_kzt'], bins=50, color='salmon', edgecolor='black')
plt.title('Распределение общей суммы транзакций (KZT)')
plt.xlabel('Общая сумма транзакций (KZT)')
plt.ylabel('Частота (количество клиентов)')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Построим точечную диаграмму зависимости общей суммы транзакций от количества транзакций
plt.figure(figsize=(10, 6))
plt.scatter(df_features['total_transactions'], df_features['total_amount_kzt'], color='green', alpha=0.6)
plt.title('Зависимость общей суммы транзакций от количества транзакций')
plt.xlabel('Количество транзакций')
plt.ylabel('Общая сумма транзакций (KZT)')
plt.grid(True)
plt.show()


# Гистограмма для количества транзакций с логарифмическим преобразованием
plt.figure(figsize=(10, 6))
log_total_transactions = np.log1p(df_features['total_transactions'])
plt.hist(log_total_transactions, bins=50, color='skyblue', edgecolor='black')
plt.title('Логарифмически преобразованное распределение количества транзакций')
plt.xlabel('log(Количество транзакций + 1)')
plt.ylabel('Частота (число клиентов)')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Гистограмма для общей суммы транзакций с логарифмическим преобразованием
plt.figure(figsize=(10, 6))
log_total_amount = np.log1p(df_features['total_amount_kzt'])
plt.hist(log_total_amount, bins=50, color='salmon', edgecolor='black')
plt.title('Логарифмически преобразованное распределение общей суммы транзакций (KZT)')
plt.xlabel('log(Общая сумма транзакций + 1)')
plt.ylabel('Частота (количество клиентов)')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Точечная диаграмма зависимости общей суммы транзакций от количества транзакций с логарифмическим преобразованием
plt.figure(figsize=(10, 6))
plt.scatter(np.log1p(df_features['total_transactions']),
            np.log1p(df_features['total_amount_kzt']),
            color='green', alpha=0.6)
plt.title('Зависимость log(Общая сумма транзакций) от log(Количество транзакций)')
plt.xlabel('log(Количество транзакций + 1)')
plt.ylabel('log(Общая сумма транзакций (KZT) + 1)')
plt.grid(True)
plt.show()

# Пути для сохранения CSV файлов (при необходимости измените пути на актуальные для вашего проекта)
output_path_features = 'data/processed/client_features.csv'
output_path_transactions = 'data/processed/transactions_processed.csv'

# Сохранение агрегированных признаков (на клиентском уровне)
df_features.to_csv(output_path_features, index=True)
print(f"Данные по агрегированным признакам сохранены в: {output_path_features}")

# Также, если вам необходимо сохранить обработанный DataFrame транзакций, выполненный на этапе 1:
df_transactions.to_csv(output_path_transactions, index=False)
print(f"Обработанные транзакционные данные сохранены в: {output_path_transactions}")

# Пути для сохранения CSV файлов (при необходимости измените пути на актуальные для вашего проекта)
output_path_features = 'data/processed/client_features.parquet'
output_path_transactions = 'data/processed/transactions_processed.parquet'

# Сохранение агрегированных признаков (на клиентском уровне)
df_features.to_parquet(output_path_features, index=True)
print(f"Данные по агрегированным признакам сохранены в: {output_path_features}")

# Также, если вам необходимо сохранить обработанный DataFrame транзакций, выполненный на этапе 1:
df_transactions.to_parquet(output_path_transactions, index=False)
print(f"Обработанные транзакционные данные сохранены в: {output_path_transactions}")