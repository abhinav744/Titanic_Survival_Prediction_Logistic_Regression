# 🚢 Titanic Survival Prediction – Logistic Regression

A classic machine learning project using Logistic Regression to predict passenger survival on the Titanic. It showcases the end-to-end pipeline from data analysis to model evaluation, highlighting the importance of feature engineering and model interpretability.

🔗 Live Demo: [Titanic Survival Prediction App](https://titanicsurvivalpredictionlogisticregression-hkl76vu2jra5ge8mb2.streamlit.app/)

## ⏳ Project Overview

Objective: Predict the likelihood of survival using passenger features such as age, gender, class, etc.

Model: Logistic Regression—ideal for binary classification and offers interpretability through coefficients.

Includes:

Data preprocessing (handling missing data, encoding categorical features)

Exploratory Data Analysis (visual insights)

Model training and evaluation (accuracy, confusion matrix, ROC-AUC)

Deployment with Streamlit

## 📂 Repository Structure

/Titanic_Survival_Prediction_Logistic_Regression

│── train.csv                         # Titanic dataset

│── titanic_logistic_regression.ipynb # Analysis & model notebook

│── app.py                            # Streamlit app

│── requirements.txt                  # Dependencies

│── README.md                         # Project documentation

## 🚀 Getting Started

### 1. Clone the Repository

git clone https://github.com/abhinav744/Titanic_Survival_Prediction_Logistic_Regression.git

cd Titanic_Survival_Prediction_Logistic_Regression

### 2. (Optional) Set Up a Virtual Environment

python -m venv venv

source venv/bin/activate       # On Windows: venv\Scripts\activate

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run the Notebook

jupyter notebook titanic_logistic_regression.ipynb

### 5. Run the Streamlit App

streamlit run app.py

## 📊 Insights & Results

Achieved classification accuracy typically around 80%

ROC-AUC scores often exceed 0.80, indicating strong model performance

Important predictors: Gender, Passenger Class, Age, and Fare

## 🔮 Future Enhancements

Add regularization (Ridge/Lasso) and hyperparameter tuning

Compare with more complex models like Random Forest or XGBoost

Visualize results: ROC curves, confusion matrices, and feature importance
