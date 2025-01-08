# =====================
# 1) Imports & Setup
# =====================
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Models
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Saving/Loading Models
import joblib

# =====================
# 2) Load and Preview Data
# =====================
pollution_data = pd.read_csv('updated_pollution_dataset.csv')
print("Initial Data (head):")
print(pollution_data.head())

# =====================
# 3) Preprocessing
# =====================

# Clean up column names for consistency
pollution_data.columns = pollution_data.columns.str.strip()

# Encoding Air Quality into numerical values
air_quality_mapping = {
    'Good': 1,
    'Hazardous': 4,
    'Moderate': 2,
    'Poor': 3
}
pollution_data['Air Quality'] = pollution_data['Air Quality'].map(air_quality_mapping)

# Selecting relevant columns for analysis
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
pollution_data_filtered = pollution_data[columns].rename(
    columns={'Proximity_to_Industrial_Areas': 'Industrial Proximity'}
)

# Optional: display basic data info
print("\nFiltered Data (head):")
print(pollution_data_filtered.head())
print("\nData Description:")
print(pollution_data_filtered.describe())

# =====================
# 4) Exploratory Analysis
# =====================

# 4.1 Calculate correlation with Air Quality
correlations = pollution_data_filtered.corr()['Air Quality'][1:]
print("\nCorrelations with Air Quality (excluding itself):")
print(correlations)

# 4.2 Scatter plots (Air Quality vs each feature)
plt.figure(figsize=(15, 10))
for i, column in enumerate(pollution_data_filtered.columns[1:], start=1):
    plt.subplot(3, 3, i)
    sns.scatterplot(
        data=pollution_data_filtered, 
        x=column, 
        y='Air Quality'
    )
    plt.title(f'Air Quality vs {column}')
    plt.xlabel(column)
    plt.ylabel('Air Quality')
plt.tight_layout()
plt.show()

# =====================
# 5) Prepare Data for Modeling
# =====================

# Optionally drop 'PM2.5' if that's intended
pollution_data_filtered = pollution_data_filtered.drop(columns=['PM2.5'])

# Create feature matrix X and target y
X = pollution_data_filtered.drop(columns=['Air Quality'])
y = pollution_data_filtered['Air Quality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)
print("\nData Shapes:")
print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"  y_train: {y_train.shape}, y_test: {y_test.shape}")

# =====================
# 6) Hyperparameter Tuning (XGBRegressor)
# =====================

# Grid of hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Initialize model
xgb_reg = XGBRegressor(random_state=42)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_reg,
    param_grid=param_grid,
    scoring='r2',
    cv=3,
    verbose=1
)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("\nBest Parameters from GridSearchCV:")
print(grid_search.best_params_)

# Evaluate on test set
y_pred_best = best_model.predict(X_test)
print("\nPerformance of the Best Model on the Test Set:")
print("  R²:", r2_score(y_test, y_pred_best))

# 6.1 Cross-validation on the best model
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
print("  Average R² (Cross-Validation):", cv_scores.mean())

# 6.2 RMSE
rmse = sqrt(mean_squared_error(y_test, y_pred_best))
print("  RMSE:", rmse)

# =====================
# 7) Compare with CatBoost
# =====================

# CatBoost Regressor
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

# XGBoost (manually configured) for comparison
xgb_reg_manual = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
xgb_reg_manual.fit(X_train, y_train)
y_pred_xgb = xgb_reg_manual.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("\nComparison:")
print("  CatBoostRegressor => MSE:", mse_cat, ", R²:", r2_cat)
print("  XGBRegressor      => MSE:", mse_xgb, ", R²:", r2_xgb)

# =====================
# 8) Feature Importances (from best XGB model)
# =====================
feature_importances = best_model.feature_importances_

# Sort feature importances
indices = np.argsort(feature_importances)[::-1]
names = [X_train.columns[i] for i in indices]

# Plot
plt.figure(figsize=(10, 6))
plt.barh(range(len(names)), feature_importances[indices], align='center')
plt.yticks(range(len(names)), names)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance - Best XGB Model')
plt.gca().invert_yaxis()
plt.show()

# =====================
# 9) Save Model & Test Loading
# =====================

joblib.dump(best_model, "xgb_regressor.pkl")
print("\nModel saved as 'xgb_regressor.pkl'.")

loaded_model = joblib.load("xgb_regressor.pkl")
print("Reloaded model successfully.")

# Test the loaded model on new data
new_data = pd.DataFrame({
    'PM10': [17.9],
    'NO2': [18.9],
    'SO2': [9.2],
    'CO': [1.72],
    'Temperature': [29.8],
    'Humidity': [59.1],
    'Industrial Proximity': [6.3]
})
prediction = loaded_model.predict(new_data)
print("\nPrediction on new data:", prediction[0])

# Reverse mapping to original Air Quality categories
reverse_mapping = {1: 'Good', 2: 'Moderate', 3: 'Poor', 4: 'Hazardous'}
predicted_category = round(prediction[0])
predicted_label = reverse_mapping.get(predicted_category, "Unknown")
print("Predicted Air Quality category:", predicted_label)

# =====================
# 10) Train Final Model on Full Data & Save
# =====================

X_full = pd.concat([X_train, X_test], axis=0)
y_full = pd.concat([y_train, y_test], axis=0)

# Use the best hyperparams found by GridSearchCV
best_params = grid_search.best_params_

final_model = XGBRegressor(
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    n_estimators=best_params['n_estimators'],
    random_state=42
)

final_model.fit(X_full, y_full)
joblib.dump(final_model, "final_xgb_model.pkl")
print("\nTrained final model on the entire dataset and saved to 'final_xgb_model.pkl'.")

y_full_pred = final_model.predict(X_full)
mse_full = mean_squared_error(y_full, y_full_pred)
r2_full = r2_score(y_full, y_full_pred)

print("Evaluation on full dataset:")
print(f"  MSE: {mse_full:.4f}")
print(f"  R²:  {r2_full:.4f}")
