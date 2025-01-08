import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ML imports
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

###############################
# ШАГ 1: Заголовок и описание
###############################
st.title("Прогноз качества воздуха (Air Quality)")

st.write("""
Демонстрация Streamlit-приложения для **регрессии** уровня загрязнения воздуха.
""")

###############################
# ШАГ 2: Загрузка и изучение данных
###############################
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

###############################
# Отображение сырых данных (больше 5 строк)
###############################
with st.expander("Данные"):
    st.write("**Сырые данные (первые 15 строк)**")
    st.dataframe(df.head(15))  # Показываем 15 строк вместо 5

###############################
# ШАГ 3: Формируем X_raw и y_raw
###############################
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

###############################
# ШАГ 4: Визуализация (Plotly)
###############################
st.subheader("Визуализация данных (Plotly)")

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

###############################
# Новый блок: Анализ зависимости между переменными (Seaborn)
###############################
import matplotlib.pyplot as plt
import seaborn as sns

if st.checkbox("Анализ зависимости между переменными (корреляции, scatter)"):
    st.write("""
    Ниже приведён код, который переименовывает столбцы, вычисляет корреляции Air Quality 
    с остальными признаками и строит scatter plots.
    """)
    
    # Создаём копию df для "pollution_data_filtered"
    pollution_data_filtered = df[[
        'Air Quality',
        'PM2.5',
        'PM10',
        'NO2',
        'SO2',
        'CO',
        'Temperature',
        'Humidity',
        'Proximity_to_Industrial_Areas'
    ]].copy()

    # Переименуем колонки для удобства
    pollution_data_filtered.columns = [
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

    # Вычисляем корреляцию (без 'Air Quality' самой)
    correlations = pollution_data_filtered.corr()['Air Quality'][1:]
    st.write("**Корреляции с Air Quality**:")
    st.write(correlations)

    # Scatter plots
    fig_corr, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, col_name in enumerate(pollution_data_filtered.columns[1:], start=1):
        ax = axes[(i - 1) // 3, (i - 1) % 3]
        sns.scatterplot(data=pollution_data_filtered, x=col_name, y='Air Quality', ax=ax)
        ax.set_title(f'Air Quality vs {col_name}')
    plt.tight_layout()
    st.pyplot(fig_corr)

###############################
# ШАГ 5: Показать новую таблицу без 'PM2.5'
###############################
if st.checkbox("Показать таблицу без столбца PM2.5"):
    # pollution_data_filtered без PM2.5
    df_no_pm25 = X_raw.drop(columns=['PM2.5']) if 'PM2.5' in X_raw.columns else X_raw
    st.write("**Таблица без столбца 'PM2.5':**")
    st.dataframe(df_no_pm25.head(15))

###############################
# ШАГ 6: Выбор модели (CatBoost, XGBoost) + разделение
###############################
st.subheader("Сравнение моделей CatBoost и XGB (без гиперпараметров)")

if st.checkbox("Запустить CatBoost/XGB без гиперпараметров"):
    from catboost import CatBoostRegressor

    # Разделение данных на train/test
    X_train0, X_test0, y_train0, y_test0 = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
    st.write(f"X_train0: {X_train0.shape}, X_test0: {X_test0.shape}")
    st.write(f"y_train0: {y_train0.shape}, y_test0: {y_test0.shape}")

    # CatBoost
    catboost_reg = CatBoostRegressor(
        iterations=100,
        depth=5,
        learning_rate=0.1,
        random_seed=42,
        verbose=False
    )
    catboost_reg.fit(X_train0, y_train0)
    y_pred_cat = catboost_reg.predict(X_test0)
    mse_cat = mean_squared_error(y_test0, y_pred_cat)
    r2_cat = r2_score(y_test0, y_pred_cat)

    # XGB
    xgb_reg0 = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    xgb_reg0.fit(X_train0, y_train0)
    y_pred_xgb = xgb_reg0.predict(X_test0)
    mse_xgb = mean_squared_error(y_test0, y_pred_xgb)
    r2_xgb = r2_score(y_test0, y_pred_xgb)

    st.write(f"**CatBoostRegressor**: MSE = {mse_cat:.4f}, R² = {r2_cat:.4f}")
    st.write(f"**XGBRegressor**: MSE = {mse_xgb:.4f}, R² = {r2_xgb:.4f}")

###############################
# ШАГ 7: Настройка гиперпараметров (GridSearchCV)
###############################
st.subheader("Обучение модели XGBRegressor (GridSearchCV)")

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

model = XGBRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

if st.button("Запустить GridSearchCV"):
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    st.write("**Лучшие гиперпараметры**:", grid_search.best_params_)

    # Оценка на тестовой выборке
    y_pred_best = best_model.predict(X_test)
    r2_best = r2_score(y_test, y_pred_best)
    st.write("**R² для лучшей модели**:", round(r2_best, 4))

    ##################################
    # Важность признаков (шаг 7)
    ##################################
    st.subheader("Важность признаков (GridSearch Best Model)")
    import matplotlib.pyplot as plt
    import numpy as np

    feature_importances = best_model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    names = [X_train.columns[i] for i in indices]

    fig_imp, ax_imp = plt.subplots(figsize=(10,6))
    ax_imp.barh(range(len(names)), feature_importances[indices], align='center')
    ax_imp.set_yticks(range(len(names)))
    ax_imp.set_yticklabels(names)
    ax_imp.set_xlabel('Важность')
    ax_imp.set_ylabel('Признаки')
    ax_imp.set_title('Важность признаков')
    ax_imp.invert_yaxis()
    st.pyplot(fig_imp)

    ##################################
    # Объединение данных и финальное обучение
    ##################################
    import joblib

    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)

    # Берем лучшие параметры
    best_params = grid_search.best_params_

    final_model = XGBRegressor(
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        n_estimators=best_params['n_estimators'],
        random_state=42
    )
    final_model.fit(X_full, y_full)

    # Сохранение модели
    joblib.dump(final_model, "final_xgb_model.pkl")
    st.write("Модель обучена на полном наборе данных и сохранена как 'final_xgb_model.pkl'.")

    # Оценка на полном наборе
    y_full_pred = final_model.predict(X_full)
    mse_full = mean_squared_error(y_full, y_full_pred)
    r2_full = r2_score(y_full, y_full_pred)
    st.write("**Оценка на полном наборе**:")
    st.write(f"MSE: {mse_full:.4f}")
    st.write(f"R² Score: {r2_full:.4f}")

    ##################################
    # 4.8 Проверка и тестирование
    ##################################
    st.write("### Проверка и тестирование (загруженной модели)")
    st.info("Заметьте, код ниже ожидает, что модель сохранена как 'xgb_regressor.pkl', но сейчас мы сохранили final_xgb_model.pkl. При желании переименуйте.")
    loaded_model = joblib.load("final_xgb_model.pkl")

    # Пример новых данных
    new_data = pd.DataFrame({
        'PM10': [17.9],
        'NO2': [18.9],
        'SO2': [9.2],
        'CO': [1.72],
        'Temperature': [29.8],
        'Humidity': [59.1],
        'Industrial Proximity': [6.3]
    })
    st.write("Пример новых данных:")
    st.dataframe(new_data)

    prediction = loaded_model.predict(new_data)[0]
    st.write("Прогнозируемое Air Quality:", prediction)

    reverse_mapping = {1: 'Good', 2: 'Moderate', 3: 'Poor', 4: 'Hazardous'}
    predicted_category = round(prediction)
    predicted_label = reverse_mapping.get(predicted_category, "Неизвестно")
    st.write(f"Округлённая категория: {predicted_label}")

else:
    st.info("Нажмите кнопку **Запустить GridSearchCV**, чтобы обучить XGBoost с подбором гиперпараметров.")
