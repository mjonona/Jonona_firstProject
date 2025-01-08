import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ML imports
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

st.title("Прогноз качества воздуха (Air Quality)")

st.write("""
Демонстрация Streamlit-приложения для **регрессии** уровня загрязнения воздуха.
""")

# ========================
# 1) Загрузка и изучение данных
# ========================
@st.cache_data
def load_data(path: str = "updated_pollution_dataset.csv"):
    df = pd.read_csv(path)
    # Чистим названия столбцов
    df.columns = df.columns.str.strip()
    # Кодируем Air Quality (если необходимо)
    air_quality_mapping = {
        'Good': 1,
        'Hazardous': 4,
        'Moderate': 2,
        'Poor': 3
    }
    if 'Air Quality' in df.columns:
        df['Air Quality'] = df['Air Quality'].map(air_quality_mapping)
    return df

df = load_data()

with st.expander("Данные"):
    st.write("**Сырые данные (первые 5 строк)**")
    st.dataframe(df.head())

# ========================
# 2) Разделяем X и y
# ========================
# Предположим, что в df есть столбцы: Air Quality, PM2.5, PM10, NO2, SO2, CO, Temperature, Humidity, Proximity_to_Industrial_Areas
# И хотим предсказывать 'Air Quality'

features = [
    'PM2.5',
    'PM10',
    'NO2',
    'SO2',
    'CO',
    'Temperature',
    'Humidity',
    'Proximity_to_Industrial_Areas'
]
target_col = 'Air Quality'

X_raw = df[features].copy()
y_raw = df[target_col].copy()

with st.expander("Преобразование данных"):
    st.write("**Признаки (X_raw)**")
    st.dataframe(X_raw.head())
    st.write("**Целевая переменная (y_raw)**")
    st.dataframe(y_raw.head())

# ========================
# 3) Визуализация данных (Plotly)
# ========================
st.subheader("Визуализация данных")

# Пример: график PM10 vs. Air Quality
fig1 = px.scatter(
    df,
    x='PM10',
    y='Air Quality',
    title='Air Quality vs. PM10'
)
st.plotly_chart(fig1)

# Пример: гистограмма PM2.5
fig2 = px.histogram(
    df,
    x='PM2.5',
    nbins=30,
    title='Распределение PM2.5'
)
st.plotly_chart(fig2)

# ========================
# 4) Пользовательский ввод (Sidebar)
# ========================
st.sidebar.header("Введите новые данные для прогноза:")

default_PM25 = float(df['PM2.5'].mean()) if 'PM2.5' in df.columns else 12.0
PM25_val = st.sidebar.number_input("PM2.5", value=default_PM25)

default_PM10 = float(df['PM10'].mean()) if 'PM10' in df.columns else 15.0
PM10_val = st.sidebar.number_input("PM10", value=default_PM10)

default_NO2 = float(df['NO2'].mean()) if 'NO2' in df.columns else 20.0
NO2_val = st.sidebar.number_input("NO2", value=default_NO2)

default_SO2 = float(df['SO2'].mean()) if 'SO2' in df.columns else 10.0
SO2_val = st.sidebar.number_input("SO2", value=default_SO2)

default_CO = float(df['CO'].mean()) if 'CO' in df.columns else 0.5
CO_val = st.sidebar.number_input("CO", value=default_CO)

default_temp = float(df['Temperature'].mean()) if 'Temperature' in df.columns else 25.0
temp_val = st.sidebar.number_input("Temperature (°C)", value=default_temp)

default_hum = float(df['Humidity'].mean()) if 'Humidity' in df.columns else 50.0
hum_val = st.sidebar.number_input("Humidity (%)", value=default_hum)

default_indus = float(df['Proximity_to_Industrial_Areas'].mean()) if 'Proximity_to_Industrial_Areas' in df.columns else 5.0
indus_val = st.sidebar.number_input("Proximity_to_Industrial_Areas", value=default_indus)

new_data = {
    'PM2.5': PM25_val,
    'PM10': PM10_val,
    'NO2': NO2_val,
    'SO2': SO2_val,
    'CO': CO_val,
    'Temperature': temp_val,
    'Humidity': hum_val,
    'Proximity_to_Industrial_Areas': indus_val
}
new_data_df = pd.DataFrame([new_data])

# ========================
# 5) Обучение модели (GridSearchCV)
# ========================
st.subheader("Обучение модели XGBRegressor")

# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# Определяем сетку гиперпараметров
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}

model = XGBRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

if st.button("Запустить GridSearchCV"):
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    st.write("**Лучшие гиперпараметры**:", grid_search.best_params_)

    # Оценка на тестовых данных
    y_pred = best_model.predict(X_test)
    r2_val = r2_score(y_test, y_pred)
    mse_val = mean_squared_error(y_test, y_pred)
    st.write(f"**R² на тестовых данных**: {r2_val:.4f}")
    st.write(f"**MSE на тестовых данных**: {mse_val:.4f}")

    # Предсказание на новых данных (из sidebar)
    pred_new = best_model.predict(new_data_df)[0]
    st.write("### Прогноз для новых данных")
    st.write("Вычисленное значение Air Quality (числовое):", round(pred_new, 2))

    # Обратное отображение (если Air Quality = 1=Good, 2=Moderate, ... )
    reverse_mapping = {1: 'Good', 2: 'Moderate', 3: 'Poor', 4: 'Hazardous'}
    cat_pred_new = round(pred_new)
    if cat_pred_new in reverse_mapping:
        st.success(f"Согласно округлению, Air Quality: **{reverse_mapping[cat_pred_new]}**")
    else:
        st.info(f"Предсказанное значение: {cat_pred_new}, не в диапазоне [1..4]")

    # Важность признаков
    feature_importances = best_model.feature_importances_
    st.write("**Важность признаков**")
    for feat, imp in zip(X_train.columns, feature_importances):
        st.write(f"{feat}: {imp:.3f}")
    
else:
    st.write("Нажмите кнопку выше, чтобы запустить поиск по сетке гиперпараметров (GridSearchCV).")
