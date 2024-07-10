import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import gradio as gr

# Load the dataset
df = pd.read_csv(r"C:\Users\hovii\OneDrive\سطح المكتب\task2day4\insurance.csv")

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

# Prediction function
def predict_insurance_charge(age, bmi, children, sex, smoker):
    user_data = pd.DataFrame(
        data=[[age, bmi, children, sex, smoker]],
        columns=['age', 'bmi', 'children', 'sex', 'smoker']
    )
    prediction = pipeline.predict(user_data)
    return f"Predicted Insurance Charge: ${prediction[0]:.2f}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_insurance_charge,
    inputs=[
        gr.Slider(label="How old are you?", minimum=18, maximum=100, value=30, step=1),
        gr.Slider(label="What is your BMI?", minimum=10.0, maximum=50.0, value=25.0, step=0.1),
        gr.Slider(label="Do you have children? If yes, how many?", minimum=0, maximum=10, value=0, step=1),
        gr.Radio(label="What is your gender?", choices=["male", "female"], value="male"),
        gr.Radio(label="Are you a smoker?", choices=["yes", "no"], value="no")
    ],
    outputs=gr.Textbox(label="Predicted Insurance Charge"),
    title="Insurance Charges Prediction",
    description="Enter your details to predict the insurance charges."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
