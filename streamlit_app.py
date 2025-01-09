import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

st.title('😁😂 My first website')

st.write('Тут я задеплою модель классификации')

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

with st.expander('Data'):
  st.write("X")
  X_raw = df.drop('species', axis=1)
  st.dataframe(X_raw)

  st.write("y")
  y_raw = df.species
  st.dataframe(y_raw)

with st.sidebar:
  st.header("Введите признаки: ")
  island = st.selectbox('Island', ('Torgersen', 'Dream', 'Biscoe'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 44.5)
  bill_depth_mm = st.slider('Bill length (mm)', 13.1, 21.5, 17.3)
  flipper_length_mm = st.slider('Flipper length (mm)', 32.1, 59.6, 44.5)
  body_mass_g = st.slider('Body mass (g)', 32.1, 59.6, 44.5)
  gender = st.selectbox('Gender', ('female', 'male'))

# Plotting some features
st.subheader('Data Visualization')
fig = px.scatter(
    df,
    x='bill_length_mm',
    y='bill_depth_mm',
    color='island',
    title='Bill Length vs. Bill Depth by Island'
)
st.plotly_chart(fig)

fig2 = px.histogram(
    df, 
    x='body_mass_g', 
    nbins=30, 
    title='Distribution of Body Mass'
)
st.plotly_chart(fig2)

## Preprocessing
data = {
    'island': island,
    'bill_length_mm': bill_length_mm,
    'bill_depth_mm': bill_depth_mm,
    'flipper_length_mm': flipper_length_mm,
    'body_mass_g': body_mass_g,
    'sex': gender
}
input_df = pd.DataFrame(data, index=[0])
input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
    st.write('**Input penguin**')
    st.dataframe(input_df)
    st.write('**Combined penguins data** (input row + original data)')
    st.dataframe(input_penguins)

encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

# Separate the top row (our input) from the rest
X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode the target
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
def target_encode(val):
    return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
    st.write('**Encoded X (input penguin)**')
    st.dataframe(input_row)
    st.write('**Encoded y**')
    st.write(y)

# Model Training
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10]
}

# Create the base model
base_rf = RandomForestClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(base_rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
st.write("**Best Parameters**:", best_params)

# ---------------------------
# 7) Apply the best model to make predictions
# ---------------------------
prediction = best_model.predict(input_row)
prediction_proba = best_model.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])

## Print model final results

st.subheader('Predicted Species')
st.dataframe(
    df_prediction_proba,
    column_config={
        'Adelie': st.column_config.ProgressColumn(
            'Adelie',
            format='%f',
            width='medium',
            min_value=0,
            max_value=1
        ),
        'Chinstrap': st.column_config.ProgressColumn(
            'Chinstrap',
            format='%f',
            width='medium',
            min_value=0,
            max_value=1
        ),
        'Gentoo': st.column_config.ProgressColumn(
            'Gentoo',
            format='%f',
            width='medium',
            min_value=0,
            max_value=1
        ),
    },
    hide_index=True
)

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(f"Predicted species: **{penguins_species[prediction][0]}**")
