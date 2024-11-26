# Home Credit Default Risk Analysis Application

This application predicts the likelihood of loan default using an XGBoost machine learning model. It includes a Flask backend, a web-based frontend, and a robust pipeline for preprocessing and modeling.


---

## Features

 1.**Machine Learning**
- Uses an XGBoost model for predictions.
- Encodes categorical data and balances classes using SMOTE.

 2.**Frontend**
- Provides a user-friendly interface to input loan details and view predictions.

 3.** Backend**
- Built with Flask, handles data preprocessing and API integration.

---

## Requirements

Ensure the following are installed on your system:

1. Python 3.12.4 (or compatible version)
2. Flask
3. XGBoost
4. Pandas
5. Scikit-learn
6. Imbalanced-learn
7. HTML/CSS for frontend design

---

## How to Run

 1.**Setup the Environment** :
Navigate to the project folder:
  ```bash
  cd application/
    Install the required Python packages:
  pip install -r requirements.txt

2.**Start the Application** :
  Run the Flask application:
  ```bash
  python app.py
The terminal will display a server link similar to:
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
3.** Access the Frontend** :
-Click the server link shown in the terminal or open it in your browser.
-Enter the required loan details and submit the form to view the prediction results.



















