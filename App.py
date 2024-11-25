from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib


best_rf_model = joblib.load('optimized_rf_model.pkl')
app = Flask(__name__)

model_features = best_rf_model.feature_names_in_

default_values ={feature: 0 for feature in model_features }

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Preprocess function to handle categorical data
def preprocess_data(input_data):
    # Example mappings for categorical features
    home_ownership_map = {'OWN': 0, 'RENT': 1, 'MORTGAGE': 2, 'OTHER': 3}
    loan_intent_map = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'DEBT_CONSOLIDATION': 3, 'HOME_IMPROVEMENT': 4, 'VENTURE': 5}
    loan_grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}

    feature_dict = default_values.copy()

    # Map frontend inputs to corresponding model features
    feature_dict.update({
        'AGE': input_data['age'],  # Map appropriately
        'AMT_INCOME_TOTAL': input_data['income'],
        'AMT_CREDIT': input_data['loan_amount'],
        'AMT_ANNUITY': input_data['loan_amount'] * 0.05,  # Approximation
        'AMT_GOODS_PRICE': input_data['loan_amount'] * 0.9,  # Approximation
        'CREDIT_HISTORY_LENGTH': input_data['credit_history_length'],  # Map example
        'HISTORICAL_DEFAULT': input_data['historical_default'],
    })

    # Categorical feature mappings
    feature_dict['home_ownership'] = home_ownership_map[input_data['home_ownership']]
    feature_dict['loan_intent'] = loan_intent_map[input_data['loan_intent']]
    feature_dict['loan_grade'] = loan_grade_map[input_data['loan_grade']]

    # Convert the dictionary to a DataFrame with the same column order as the model
    processed_data = pd.DataFrame([feature_dict])[model_features]
    return processed_data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        print("Raw Input Data:", input_data)  # Log input data
        processed_data = preprocess_data(input_data)
        print("Processed Data:", processed_data)  # Log processed data
        prediction = best_rf_model.predict(processed_data)
        print("Prediction:", prediction)  # Log prediction
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
