from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd


app = Flask(__name__)

# Function to calculate credit score
def calculate_credit_score(input_data):
    # 1. Payment History Score
    sk_dpd = input_data.get('SK_DPD', 0)
    amt_payment = input_data.get('AMT_PAYMENT', 0)
    amt_installment = input_data.get('AMT_INSTALMENT', 0)
    payment_ratio = amt_payment / amt_installment if amt_installment > 0 else 1
    payment_history_score = max(0, min(1, payment_ratio) * (1 - sk_dpd / 100)) * 0.35

    # 2. Credit Utilization Score
    amt_balance = input_data.get('AMT_BALANCE', 0)
    amt_credit = input_data.get('AMT_CREDIT', 1)  # Avoid division by zero
    credit_utilization_ratio = amt_balance / amt_credit if amt_credit > 0 else 1
    credit_utilization_score = max(0, 1 - credit_utilization_ratio) * 0.3

    # 3. Length of Credit History Score
    days_credit = input_data.get('DAYS_CREDIT', 0)
    days_decision = input_data.get('DAYS_DECISION', 0)
    if days_credit and days_decision:
        credit_history_length = abs(days_credit)  # Use absolute since values are negative
    else:
        credit_history_length = 0  # Default if either is missing
    length_of_credit_history_score = min(1, credit_history_length / 3650) * 0.15

    # 4. Credit Mix Score
    name_contract_type = input_data.get('NAME_CONTRACT_TYPE', 'Unknown')
    credit_mix_score = 0.1 if name_contract_type in ['Cash loans', 'Revolving loans'] else 0.05

    # 5. New Credit Score
    amt_application = input_data.get('AMT_APPLICATION', 0)
    new_credit_score = max(0, 1 - (amt_application / 500000)) * 0.1

    # Calculate Total Credit Score
    total_credit_score = (payment_history_score + credit_utilization_score +
                          length_of_credit_history_score + credit_mix_score + new_credit_score) * 850

    return int(total_credit_score)

# Determine FICO range
def determine_fico_range(credit_score):
    if credit_score >= 800:
        return "Exceptional"
    elif credit_score >= 740:
        return "Very Good"
    elif credit_score >= 669:
        return "Good"
    elif credit_score >= 580:
        return "Fair"
    else:
        return "Poor"


# Load the pipeline
pipeline = joblib.load('C:\\Users\\SMRUTI DESHPANDE\\house credit default\\credit_model_pipeline.pkl')
app.logger.info("Pipeline loaded successfully.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {k: float(v) if k in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_BALANCE', 
                                           'AMT_ANNUITY', 'SK_DPD', 'CNT_CHILDREN', 
                                           'DAYS_CREDIT', 'DAYS_DECISION', 'AMT_PAYMENT', 
                                           'AMT_INSTALMENT', 'AMT_APPLICATION'] 
                      else v 
                      for k, v in request.form.items()}
   
        input_df = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)[0]
        
        credit_score = calculate_credit_score(input_data)
        fico_range = determine_fico_range(credit_score)
        if fico_range in ['Poor', 'Fair']:
            prediction = 1 
        
        # Return results
        result = {
            'prediction': 'Defaulter' if prediction == 1 else 'Non-Defaulter',
            'creditScore': credit_score,
            'ficoRange': fico_range
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

    
if __name__ == '__main__':
    app.run(debug=True)