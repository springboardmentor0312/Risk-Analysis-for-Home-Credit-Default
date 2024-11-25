from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load pre-trained models
random_forest_model = joblib.load("models/random_forest_model.pkl")
decision_tree_model = joblib.load("models/decision_tree_model.pkl")

# Load the column names used during training
expected_columns = joblib.load("models/training_columns.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age_category = request.form['age_category']
        loan_type = request.form['loan_type_radio']
        payment_count = int(request.form['payment_count'])
        total_amount_paid = float(request.form['total_amount_paid'])
        average_payment_delay = float(request.form['average_payment_delay'])
        income = float(request.form['income'])
        credit = float(request.form['credit'])
        model_choice = request.form['model_choice']
        
        # One-hot encoding for age_category (3 values)
        age_category_encoded = [0, 0, 0]
        if age_category == 'young':
            age_category_encoded[0] = 1
        elif age_category == 'middle-aged':
            age_category_encoded[1] = 1
        else:  # elderly
            age_category_encoded[2] = 1

        # One-hot encoding for loan_type (2 values only)
        loan_type_encoded = [0, 0]
        if loan_type == 'cash':
            loan_type_encoded[0] = 1
        elif loan_type == 'revolving':
            loan_type_encoded[1] = 1

        # Combine all features into a single dictionary
        features_dict = {
            "age_category_young": age_category_encoded[0],
            "age_category_middle-aged": age_category_encoded[1],
            "age_category_elderly": age_category_encoded[2],
            "loan_type_cash": loan_type_encoded[0],
            "loan_type_revolving": loan_type_encoded[1],
            "payment_count": payment_count,
            "total_amount_paid": total_amount_paid,
            "average_payment_delay": average_payment_delay,
            "amt_income_total": income,
            "amt_credit": credit,
        }

        # Convert to DataFrame with expected columns
        features_df = pd.DataFrame([features_dict], columns=expected_columns).fillna(0)

        # Select the model
        model = random_forest_model if model_choice == 'random_forest' else decision_tree_model

        # Enhanced credit score calculation
        base_score = 300
        payment_history_score = 100 if payment_count > 10 and average_payment_delay <= 10 else 50
        credit_utilization_ratio = min(credit / income, 1)
        credit_utilization_score = 150 * (1 - credit_utilization_ratio)
        credit_history_score = min(payment_count * 10, 100)
        average_delay_penalty = max(50 - average_payment_delay, 0)

        # Final credit score
        raw_score = (base_score + payment_history_score + credit_utilization_score +
                     credit_history_score + average_delay_penalty)
        credit_score = int((raw_score / 700) * 550 + 300)  # Scale to a range of 300–850

        # Determine credit category based on score
        if credit_score >= 800:
            category = "Excellent"
            status = "Non-Default"
        elif credit_score >= 740:
            category = "Very Good"
            status = "Non-Default"
        elif credit_score >= 670:
            category = "Good"
            status = "Non-Default"
        elif credit_score >= 650:
            category = "Not-Bad"
            status = "Default"
        else:
            category = "Poor"
            status = "Default"

        # Reasons for default or non-default
        if status == "Default":
            reason = "High debt-to-income ratio or irregular payments observed."
            recommendation = "Consider lowering debt levels, maintaining regular payments, and avoiding new debt to improve financial health."
        else:
            reason = "Consistent payments and a balanced debt-to-income ratio."
            recommendation = "Continue maintaining financial discipline, make payments on time, and avoid taking excessive debt."

        return render_template(
            'result.html', 
            status=status, 
            score=credit_score, 
            category=category, 
            reason=reason, 
            recommendation=recommendation
        )

if __name__ == '__main__':
    app.run(debug=True)
