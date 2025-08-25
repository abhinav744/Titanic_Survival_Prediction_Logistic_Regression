# Titanic Survival Prediction App
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load and preprocess dataset
titanic_data = pd.read_csv("train (1).csv")

# Handle missing values
titanic_data = titanic_data.drop(columns="Cabin", axis=1)
titanic_data["Age"].fillna(titanic_data["Age"].mean(), inplace=True)
titanic_data["Embarked"].fillna(titanic_data["Embarked"].mode()[0], inplace=True)

# Encode categorical values
titanic_data.replace(
    {"Sex": {"male": 0, "female": 1}, "Embarked": {"S": 0, "C": 1, "Q": 2}},
    inplace=True,
)

# Features and target
X = titanic_data.drop(columns=["PassengerId", "Name", "Ticket", "Survived"], axis=1)
Y = titanic_data["Survived"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model training
model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)

# Streamlit UI
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival:")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=8, value=0)
parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=6, value=0)
fare = st.number_input("Ticket Fare", min_value=0.0, max_value=600.0, value=32.0, step=0.1)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Convert input into model format
sex_val = 0 if sex == "Male" else 1
embarked_val = {"S": 0, "C": 1, "Q": 2}[embarked]

input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 The passenger would have SURVIVED!")
    else:
        st.error("❌ The passenger would NOT have survived.")
