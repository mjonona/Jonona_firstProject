import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

###############################
 ЧАСТЬ 1. ПРЕЗЕНТАЦИЯ
###############################

st.title("Презентация проекта: Прогноз Air Quality")

# 1) Коротко о себе
st.subheader("1. Коротко о себе")
st.write("""
Меня зовут [Ваше имя]. Я — QA Engineer в компании «Согд Дизайн». 
Моя основная задача — тестировать веб-приложение **Simple EMR for personal injury clinics** 
(система для медицинских клиник по травматологии) и следить за качеством продукта.
""")

# 2) Тема проекта
st.subheader("2. Тема проекта и польза")
st.write("""
**Тема**: Прогноз качества воздуха (Air Quality). 

Почему выбрала:
- В Душанбе часто наблюдается низкое качество воздуха, и мне интересно изучить, 
  какие факторы (PM2.5, PM10, NO2, SO2, CO, температура, влажность, близость к промышленным зонам и т.д.) 
  сильнее всего влияют на итоговый показатель Air Quality.
- Данные для обучения взяты с Kaggle (примерный набор с показателями загрязнения).
- Такой проект может помочь лучше понять, какие загрязнители наиболее опасны для здоровья людей в городе.
""")

# 3) Бейзлайн и метрика успеха
st.subheader("3. Бейзлайн и метрики")
st.write("""
- **Бейзлайн**:
  - В качестве таргета берём показатель Air Quality 
  - Цель — проверить, что наша модель даёт более точный прогноз

- **Метрики**:
  - **R² (коэффициент детерминации)**: чем ближе к 1, тем лучше модель объясняет вариацию качества воздуха.
  - **MSE (среднеквадратичная ошибка)** или **RMSE**: чем ниже, тем лучше; 
    показывает, насколько сильно модель ошибается в среднем.

- **Успешные значения**:
  - R² > 0.8 часто считается хорошим показателем для регрессии в задачах экологии.
  - Чем ниже MSE, тем лучше. Конкретный порог зависит от диапазона данных по загрязнению.
""")

# 4) Проделанные шаги
st.subheader("4. Проделанные шаги")
st.write("""
1) **Сбор и очистка данных**: 
   - Взяла датасет с Kaggle, 
   - Удалила пропуски и аномальные значения (если были).
2) **Анализ признаков**:
   - Посмотрела, какие показатели (PM2.5, PM10, NO2 и т.п.) сильнее связаны с Air Quality.
3) **Разделение на train/test**:
   - Применяла `train_test_split` для оценки качества модели.
4) **Обучение моделей**:
   - Пробовала XGBoost, настраивала гиперпараметры через `GridSearchCV`.
5) **Оценка результатов**:
   - На основе R², MSE, и выбрала лучшую модель.
6) **Финальный вывод**:
   - Создала интерактивное демо-приложение в Streamlit, где пользователь может ввести свои параметры загрязнения 
     и получить мгновенный прогноз Air Quality.
""")

# 5) Результат и выводы
st.subheader("5. Результаты и выводы")
st.write("""
- Итог: Модель обучена на полном наборе данных и сохранена как 'final_xgb_model.pkl'.
Оценка модели на полном наборе данных:
Mean Squared Error (MSE): 0.0170
R² Score: 0.9830
- В перспективе можно расширить датасет реальными данными из Душанбе, 
  чтобы адаптировать модель к локальным особенностям.
""")

st.markdown("---")
st.header("ДЕМО МОДЕЛИ (XGBRegressor + GridSearchCV)")

###############################
#  ЧАСТЬ 2. DEMO МОДЕЛИ
###############################

# Загрузка данных
@st.cache_data
def load_data(path: str = "updated_pollution_dataset.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Предположим, в столбце 'Air Quality' уже числовые значения (1..4) 
    return df

df = load_data()

if st.checkbox("Показать сырые данные"):
    st.write(df.head())

# Выберем нужные столбцы
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

# Фильтруем, если все есть в датасете
available_cols = [c for c in columns if c in df.columns]
df_filtered = df[available_cols].copy()

# Переименуем для удобства (необязательно)
df_filtered.columns = [
    'Air Quality',
    'PM2.5',
    'PM10',
    'NO2',
    'SO2',
    'CO',
    'Temperature',
    'Humidity',
    'Industrial Proximity'
]

# Удалим при желании PM2.5 — если вы так хотели в предыдущих кодах (необязательно)
# df_filtered = df_filtered.drop(columns=['PM2.5'])

# Разделяем X и y
X = df_filtered.drop(columns=['Air Quality'])
y = df_filtered['Air Quality']

# Отображаем корреляции
if st.checkbox("Показать корреляции c Air Quality"):
    corr_series = df_filtered.corr()['Air Quality'][1:]
    st.write(corr_series)

    # Графики разброса (scatter plots)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, col_name in enumerate(X.columns, start=1):
        ax = axes[(i - 1) // 3, (i - 1) % 3]
        sns.scatterplot(data=df_filtered, x=col_name, y='Air Quality', ax=ax)
        ax.set_title(f'Air Quality vs {col_name}')
    plt.tight_layout()
    st.pyplot(fig)

# Разделяем на train и test
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,
    random_state=42
)

# Создаем сетку гиперпараметров
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}

xgb_reg = XGBRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_reg,
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
    rmse_val = sqrt(mse_val)

    st.write(f"**R²** на тестовых данных: {r2_val:.4f}")
    st.write(f"**MSE**: {mse_val:.4f}")
    st.write(f"**RMSE**: {rmse_val:.4f}")

    # Важность признаков
    st.subheader("Важность признаков (Best XGB Model)")
    feature_importances = best_model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_feats = X_train.columns[sorted_idx]

    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(sorted_feats, feature_importances[sorted_idx])
    ax_imp.set_title("Feature Importances")
    ax_imp.invert_yaxis()
    st.pyplot(fig_imp)

    # Предсказание на новых данных
    st.subheader("Прогноз (введите новые значения)")
    input_data = {}
    for col_name in X.columns:
        default_val = float(X[col_name].mean())
        input_data[col_name] = st.number_input(f"{col_name}", value=default_val)

    new_data_df = pd.DataFrame([input_data])
    if st.button("Сделать прогноз"):
        pred_val = best_model.predict(new_data_df)[0]
        st.write(f"**Предсказанный Air Quality**: {pred_val:.2f}")

        # Пример обратной карты (если 1=Good, 2=Moderate, 3=Poor, 4=Hazardous)
        reverse_mapping = {1: 'Good', 2: 'Moderate', 3: 'Poor', 4: 'Hazardous'}
        cat_pred = round(pred_val)
        cat_label = reverse_mapping.get(cat_pred, "Неизвестно")
        st.write(f"**Категория (по округлению)**: {cat_label}")
else:
    st.info("Нажмите кнопку выше, чтобы запустить перебор гиперпараметров (GridSearchCV) и построить модель.")
