import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# -------------------------
# 1) Заголовок приложения
# -------------------------
st.title("Прогноз качества воздуха")

# -------------------------
# 2) Загрузка данных
# -------------------------
@st.cache_data
def load_data(path: str = "updated_pollution_dataset.csv") -> pd.DataFrame:
    """
    Загружает CSV-файл, чистит названия столбцов и кодирует признак 'Air Quality'.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    
    # Кодирование категорий Air Quality
    air_quality_mapping = {
        'Good': 1,
        'Hazardous': 4,
        'Moderate': 2,
        'Poor': 3
    }
    df['Air Quality'] = df['Air Quality'].map(air_quality_mapping)
    
    return df

pollution_data = load_data()

# -------------------------
# 3) Отображение данных
# -------------------------
if st.checkbox("Показать данные"):
    st.write("**Данные (первые 5 строк):**")
    st.write(pollution_data.head())

# -------------------------
# 4) Подготовка данных
# -------------------------
# Определяем нужные столбцы
columns = [
    'Air Quality',
    'PM2.5',
    'PM10',
    'NO2',
    'SO2',
    'CO',
    'Temperature',
    'Humidity',
    'Proximity_to_Industrial_Areas'
]

# Выбираем нужные столбцы и переименовываем для удобства
pollution_data_filtered = pollution_data[columns].rename(
    columns={'Proximity_to_Industrial_Areas': 'Industrial Proximity'}
)

# -------------------------
# 5) Корреляции и визуализация
# -------------------------
if st.checkbox("Показать корреляции"):
    st.subheader("Корреляции с признаком 'Air Quality'")
    correlations = pollution_data_filtered.corr()['Air Quality'][1:]
    st.write(correlations)

    # Построение scatter plot для каждого признака vs Air Quality
    st.subheader("Взаимосвязь между признаками")
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    # Перебираем все столбцы, кроме Air Quality
    for i, column in enumerate(pollution_data_filtered.columns[1:], start=1):
        ax = axes[(i - 1) // 3, (i - 1) % 3]
        sns.scatterplot(
            data=pollution_data_filtered,
            x=column, 
            y='Air Quality',
            ax=ax
        )
        ax.set_title(f'Air Quality vs {column}')
    st.pyplot(fig)

# -------------------------
# 6) Формирование X, y
# -------------------------
# Удаляем 'PM2.5' (по условию кода) и 'Air Quality' из признаков
X = pollution_data_filtered.drop(columns=['Air Quality', 'PM2.5'])
y = pollution_data_filtered['Air Quality']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

# -------------------------
# 7) Обучение моделей (XGB + GridSearchCV)
# -------------------------
st.subheader("Обучение моделей")

# Сетка гиперпараметров
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Инициализация XGB
xgb_reg = XGBRegressor(random_state=42)

# GridSearchCV
st.write("Обучение XGBRegressor...")
grid_search = GridSearchCV(
    estimator=xgb_reg, 
    param_grid=param_grid, 
    scoring='r2', 
    cv=3, 
    verbose=0
)
grid_search.fit(X_train, y_train)

# Лучшая модель и её результаты
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
st.write(f"Лучшие параметры: {best_params}")

y_pred_best = best_model.predict(X_test)
r2_best = r2_score(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = sqrt(mse_best)

st.write(f"R² для лучшей модели: {r2_best:.4f}")
st.write(f"MSE для лучшей модели: {mse_best:.4f}")
st.write(f"RMSE для лучшей модели: {rmse_best:.4f}")

# -------------------------
# 8) Кросс-валидация
# -------------------------
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
st.write(f"Средний R² на кросс-валидации: {cv_scores.mean():.4f}")

# -------------------------
# 9) Сравнение CatBoost и XGB
# -------------------------
st.subheader("Сравнение моделей CatBoost и XGB")

# CatBoost
catboost_reg = CatBoostRegressor(
    iterations=100,
    depth=5,
    learning_rate=0.1,
    random_seed=42,
    verbose=False
)
catboost_reg.fit(X_train, y_train)
y_pred_cat = catboost_reg.predict(X_test)
mse_cat = mean_squared_error(y_test, y_pred_cat)
r2_cat = r2_score(y_test, y_pred_cat)

# XGB (с ручными параметрами)
xgb_reg_comp = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
xgb_reg_comp.fit(X_train, y_train)
y_pred_xgb = xgb_reg_comp.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

st.write("**CatBoostRegressor**:")
st.write(f"MSE: {mse_cat:.4f}, R²: {r2_cat:.4f}")
st.write("**XGBRegressor**:")
st.write(f"MSE: {mse_xgb:.4f}, R²: {r2_xgb:.4f}")

# -------------------------
# 10) Важность признаков
# -------------------------
st.subheader("Важность признаков (по лучшей XGB-модели)")
feature_importances = best_model.feature_importances_
indices = np.argsort(feature_importances)[::-1]
names = [X_train.columns[i] for i in indices]

fig, ax = plt.subplots()
ax.barh(names, feature_importances[indices])
ax.set_title("Важность признаков")
ax.invert_yaxis()  # чтобы самый важный был сверху
st.pyplot(fig)

# -------------------------
# 11) Предсказание новых данных
# -------------------------
st.subheader("Предсказание новых данных")

new_data = {
    'PM10': st.number_input("PM10", value=17.9),
    'NO2': st.number_input("NO2", value=18.9),
    'SO2': st.number_input("SO2", value=9.2),
    'CO': st.number_input("CO", value=1.72),
    'Temperature': st.number_input("Temperature", value=29.8),
    'Humidity': st.number_input("Humidity", value=59.1),
    'Industrial Proximity': st.number_input("Industrial Proximity", value=6.3)
}
new_data_df = pd.DataFrame([new_data])

if st.button("Сделать прогноз"):
    prediction = best_model.predict(new_data_df)[0]
    reverse_mapping = {1: 'Good', 2: 'Moderate', 3: 'Poor', 4: 'Hazardous'}
    predicted_category = round(prediction)
    
    if predicted_category in reverse_mapping:
        st.write(f"Прогнозируемое качество воздуха: **{reverse_mapping[predicted_category]}**")
    else:
        st.write("Прогнозируемое качество воздуха за пределами известных категорий.")
