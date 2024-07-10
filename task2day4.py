import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import streamlit as st

# Load the dataset
df = pd.read_csv("insurance.csv")  # Using relative path

# Drop the 'region' column
df = df.drop(columns=['region'])

# Separate features and target variable
X = df.drop(columns=['charges'])
y = df['charges']

# Convert categorical variables to numerical using OneHotEncoder
categorical_features = ['sex', 'smoker']
numerical_features = ['age', 'bmi', 'children']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create the Linear Regression model
linear_regression = LinearRegression()

# Create a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', linear_regression)
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Streamlit UI
st.title("Insurance Charges Prediction")

# User inputs
age = st.number_input("How old are you?", min_value=18, max_value=100, value=30)
bmi = st.number_input("What is your BMI?", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Do you have children? If yes, how many?", min_value=0, max_value=10, value=0)
sex = st.selectbox("What is your gender?", options=["male", "female"])
smoker = st.selectbox("Are you a smoker?", options=["yes", "no"])

# Predict button
if st.button("Predict"):
    user_data = pd.DataFrame(
        data=[[age, bmi, children, sex, smoker]],
        columns=['age', 'bmi', 'children', 'sex', 'smoker']
    )

    # Preprocess user data
    user_pred = pipeline.predict(user_data)
    st.write(f"Predicted Insurance Charge: ${user_pred[0]:.2f}")
