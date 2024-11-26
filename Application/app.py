from flask import Flask, request, jsonify, render_template_string

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load or train model
MODEL_PATH = 'xgb_model.pkl'
LABEL_ENCODERS_PATH = 'label_encoders.pkl'
DATASET_PATH = 'C:\INFOSYS(HCD)\HCD(Preprocessed) (1).csv'  # Set your dataset path here

# Load the dataset, preprocess, and train the model if not saved
def load_and_train_model():
    # Check if the model and encoders exist
    try:
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(LABEL_ENCODERS_PATH, 'rb') as le_file:
            label_encoders = pickle.load(le_file)
        return model, label_encoders

    except FileNotFoundError:
        print("Model or encoders not found. Training model...")

        # Load dataset
        data = pd.read_csv(DATASET_PATH)
        data['Current_loan_status'] = data['Current_loan_status'].replace({'NO DEFAULT': 0, 'DEFAULT': 1})
        data['historical_default'] = data['historical_default'].replace({'N': 'temp', 'Unknown': 'N'}).replace({'temp': 'Unknown'})

        # Encode categorical features
        label_encoders = {}
        for column in data.select_dtypes(include=['object']).columns:
            if column != 'Current_loan_status':
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
                label_encoders[column] = le

        X = data.drop('Current_loan_status', axis=1)
        y = data['Current_loan_status']

        # Handle class imbalance with SMOTE
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

        # Train the model
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)

        # Save the model and encoders
        with open(MODEL_PATH, 'wb') as model_file:
            pickle.dump(model, model_file)
        with open(LABEL_ENCODERS_PATH, 'wb') as le_file:
            pickle.dump(label_encoders, le_file)

        return model, label_encoders

# Load model and encoders
xgb_model, label_encoders = load_and_train_model()

# Credit score calculation function (same as provided)
def calculate_credit_score(user_data):
    X = 300
    payment_history_score = 200 if user_data['historical_default'] == 'N' else 50
    credit_utilization_ratio = min(user_data['loan_amnt'] / user_data['customer_income'], 1)
    credit_utilization_score = 150 * (1 - credit_utilization_ratio)
    credit_history_score = min(user_data['cred_hist_length'] * 20, 100)
    employment_stability_score = min(int(user_data['employment_duration']) / 12 * 10, 100)
    loan_grade_scores = {'A': 100, 'B': 80, 'C': 60, 'D': 40, 'E': 20, 'F': 10}
    loan_grade_score = loan_grade_scores.get(user_data['loan_grade'], 10)

    raw_credit_score = X + (payment_history_score + credit_utilization_score + credit_history_score + employment_stability_score + loan_grade_score)
    min_raw_score = 360
    max_raw_score = 950
    credit_score = int(((raw_credit_score - min_raw_score) / (max_raw_score - min_raw_score)) * 550 + 300)

    if credit_score >= 800:
        category = 'Excellent'
    elif credit_score >= 740:
        category = 'Very Good'
    elif credit_score >= 670:
        category = 'Good'
    elif credit_score >= 580:
        category = 'Fair'
    else:
        category = 'Poor'

    return credit_score, category

# Prediction function using the trained model
def predict_loan_default_and_score(user_data):
    input_data = pd.DataFrame([user_data])

    # Encode categorical variables
    for column in label_encoders:
        if column in input_data.columns:
            if input_data[column].iloc[0] not in label_encoders[column].classes_:
                input_data[column] = np.nan
            else:
                input_data[column] = label_encoders[column].transform(input_data[column])
        else:
            input_data[column] = np.nan

    input_data = input_data.fillna(0)
    input_data = input_data.astype(float)

    prediction = xgb_model.predict(input_data)
    prediction_result = 'DEFAULT' if prediction[0] == 1 else 'NO DEFAULT'

    credit_score, credit_category = calculate_credit_score(user_data)

    return prediction_result, credit_score, credit_category

# Route for the HTML page
@app.route('/')
def home():
    # Read HTML content from index.html
    with open('index.html', 'r') as file:
        html_content = file.read()
    # Pass the HTML content directly to render_template_string
    return render_template_string(html_content)

# API route to handle AJAX POST requests
@app.route('/predict', methods=['POST'])
def predict():
    user_data = {
        'customer_id': int(request.form['customer_id']),
        'customer_age': int(request.form['customer_age']),
        'customer_income': float(request.form['customer_income']),
        'home_ownership': request.form['home_ownership'],
        'employment_duration': request.form['employment_duration'],
        'loan_intent': request.form['loan_intent'],
        'loan_grade': request.form['loan_grade'],
        'loan_amnt': float(request.form['loan_amnt']),
        'loan_int_rate': float(request.form['loan_int_rate']),
        'term_years': int(request.form['term_years']),
        'historical_default': request.form['historical_default'],
        'cred_hist_length': int(request.form['cred_hist_length']),
    }

    prediction_result, credit_score, credit_category = predict_loan_default_and_score(user_data)

    return jsonify(prediction=prediction_result, credit_score=credit_score, credit_category=credit_category)

if __name__ == '__main__':
    app.run(debug=True)
