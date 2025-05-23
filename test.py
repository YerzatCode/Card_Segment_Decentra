import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")

data_dict = pd.DataFrame({
    "Название фичи": df.columns,
    "Тип данных": [str(dtype) for dtype in df.dtypes],
    "Описание": "",  # Заполни вручную или частично через код
    "Пример значения": df.iloc[0].values
})

data_dict.to_excel("data_dictionary_test.xlsx", index=False)
print("✅ Data Dictionary сохранён в data_dictionary.xlsx")
