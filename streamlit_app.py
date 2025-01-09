import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

###############################
# 1. Заголовок приложения и презентация
###############################
st.title("Презентация проекта: Прогноз Air Quality")

# 1) Коротко о себе
st.subheader("1. Коротко о себе")
st.write("""
Меня зовут Джонона. Я — QA Engineer в компании «Согд Дизайн». 
Тестирую веб-приложение **Simple EMR for personal injury clinics** 
(система для медицинских клиник по травматологии) и отвечаю за качество продукта.
""")

# 2) Тема проекта и польза
st.subheader("2. Тема проекта и польза")
st.write("""
**Тема**: Прогноз качества воздуха (Air Quality) на основе данных (Kaggle).

Почему выбрала:
- В Душанбе низкое качество воздуха, хотела понять, какие факторы (PM2.5, PM10, NO2, SO2, CO и т.д.)
  сильнее всего влияют на итоговый показатель Air Quality.
- Такой проект может быть полезен для принятия решений городскими службами, жителями, врачами.

**Полезность**:
- Позволяет ввести показатели загрязнения и получить прогноз, какие дни опасны для здоровья.
""")

# 3) Бейзлайн и метрика успеха
st.subheader("3. Бейзлайн и метрики")
st.write("""
- **Бейзлайн**:
  - Можно брать среднее или предыдущее значение Air Quality, чтобы сравнить модель с этим наивным подходом.
- **Метрики**:
  - **R² (коэффициент детерминации)**: ближе к 1 — лучше.
  - **MSE** (среднеквадратичная ошибка): чем меньше, тем модель точнее предсказывает.
- **Успешные значения**:
  - R² > 0.8 обычно считается неплохим результатом в задачах экологии.
  - MSE — чем ближе к 0, тем лучше, учитывая масштаб данных.
""")

# 4) Проделанные шаги
st.subheader("4. Проделанные шаги")
st.write("""
1. Сбор и очистка данных: (Kaggle).  
2. Анализ признаков: проверили корреляции, определили важность PM2.5, PM10, NO2 и т.д.  
3. Разделение на train/test: `train_test_split`.  
4. Обучение XGBoost (и CatBoost) с `GridSearchCV` для подбора гиперпараметров.  
5. Оценка: сравнение R², MSE с бейзлайном.  
6. Вывод: Streamlit-приложение, где пользователь может делать прогноз для новых значений.
""")

# 5) Результаты и выводы
st.subheader("5. Результаты и выводы")
st.write("""
- Модель даёт высокое R² (например, ~0.85–0.90) на тестовых данных и заметно превосходит наивный подход.
- Наиболее важные факторы: PM2.5, PM10, NO2.
- Расширение датасета реальными данными из Душанбе может повысить точность и сделать модель более прикладной.
""")

st.markdown("---")

###############################
# 6. ДЕМО МОДЕЛИ
###############################
st.header("ДЕМО МОДЕЛИ: XGBRegressor + GridSearchCV")

@st.cache_data
def load_data(path: str = "updated_pollution_dataset.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Кодируем Air Quality (если ещё не закодировано)
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

# Отображаем часть данных
with st.expander("Данные"):
    st.write("**Сырые данные (первые 10 строк)**")
    st.dataframe(df.head(10))

# Формируем X, y
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
    st.write("**Признаки (X_raw) (первые 5)**")
    st.dataframe(X_raw.head())
    st.write("**Целевая переменная (y_raw) (первые 5)**")
    st.dataframe(y_raw.head())

st.subheader("Визуализация (Plotly)")

# Пример графика PM10 vs. Air Quality
fig1 = px.scatter(
    df,
    x='PM10',
    y='Air Quality',
    title='Air Quality vs. PM10'
)
st.plotly_chart(fig1)

fig2 = px.histogram(
    df,
    x='PM2.5',
    nbins=30,
    title='Распределение PM2.5'
)
st.plotly_chart(fig2)

###############################
# Сайдбар: ввод новых значений
###############################
st.sidebar.header("Введите новые данные для прогноза Air Quality:")

default_PM25 = float(df['PM2.5'].mean()) if 'PM2.5' in df.columns else 10.0
PM25_val = st.sidebar.number_input("PM2.5", value=default_PM25)

default_PM10 = float(df['PM10'].mean()) if 'PM10' in df.columns else 15.0
PM10_val = st.sidebar.number_input("PM10", value=default_PM10)

default_NO2 = float(df['NO2'].mean()) if 'NO2' in df.columns else 20.0
NO2_val = st.sidebar.number_input("NO2", value=default_NO2)

default_SO2 = float(df['SO2'].mean()) if 'SO2' in df.columns else 8.0
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

###############################
# Обучение (GridSearchCV)
###############################
st.subheader("Обучение модели (GridSearchCV для XGBRegressor)")

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}
base_model = XGBRegressor(random_state=42)
grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='r2', n_jobs=-1)

if st.button("Запустить GridSearchCV"):
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    st.write("**Лучшие гиперпараметры**:", grid_search.best_params_)

    # Оценка на тестовой выборке
    y_pred = best_model.predict(X_test)
    r2_val = r2_score(y_test, y_pred)
    mse_val = mean_squared_error(y_test, y_pred)
    st.write(f"**R² на тестовых данных**: {r2_val:.4f}")
    st.write(f"**MSE на тестовых данных**: {mse_val:.4f}")

    # Прогноз для новых данных
    pred_new = best_model.predict(new_data_df)[0]
    st.write("### Прогноз для новых данных")
    st.write("Числовое значение Air Quality:", round(pred_new, 2))

    reverse_mapping = {1: 'Good', 2: 'Moderate', 3: 'Poor', 4: 'Hazardous'}
    cat_pred_new = round(pred_new)
    if cat_pred_new in reverse_mapping:
        st.success(f"Согласно округлению, Air Quality: **{reverse_mapping[cat_pred_new]}**")
    else:
        st.info(f"Предсказанное значение: {cat_pred_new}, не в диапазоне [1..4]")

    # Важность признаков
    st.write("**Важность признаков (Best Model)**")
    feature_importances = best_model.feature_importances_
    for feat, imp in zip(X_train.columns, feature_importances):
        st.write(f"{feat}: {imp:.3f}")
else:
    st.info("Нажмите кнопку, чтобы запустить GridSearchCV и получить прогноз.")

st.write("---")
st.write("© 2023. Мой Streamlit-проект по прогнозу Air Quality.")
