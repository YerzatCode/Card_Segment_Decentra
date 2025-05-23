import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from main import run_pipeline

st.title("📊 Поведенческая сегментация клиентов")

if st.button("🔁 Запустить кластеризацию"):
    run_pipeline()
    st.success("Кластеризация завершена!")

df = pd.read_parquet("data/processed/features_with_clusters.parquet")

st.header("📋 Сводка по кластерам")
st.dataframe(df.groupby("cluster_name").mean(numeric_only=True).round(2))

st.header("📊 Распределение по сегментам")
fig, ax = plt.subplots()
sns.countplot(data=df, x="cluster_name", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

st.header("👥 Таблица клиентов")
cluster = st.selectbox("Выберите кластер", df['cluster_name'].unique())
st.dataframe(df[df['cluster_name'] == cluster].head(20))
