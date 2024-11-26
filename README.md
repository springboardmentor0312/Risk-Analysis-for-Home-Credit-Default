Home Credit Default Risk Analysis Application
This application predicts the likelihood of loan default using an XGBoost machine learning model. It includes a Flask backend, a web-based frontend, and a robust pipeline for preprocessing and modeling.

Project Structure
Application.zip
├── app/
│   ├── static/            # Static files (CSS, JS, Images)
│   ├── templates/         # HTML files (Frontend)
│   ├── model/             # Trained models and encoders (xgb_model.pkl, label_encoders.pkl)
│   ├── app.py             # Main Flask application
│   ├── preprocess.py      # Data preprocessing utilities
│   └── requirements.txt   # Python dependencies
├── dataset/
│   └── HCD(Preprocessed).csv  # Preprocessed dataset
└── README.md              # Project documentation

Features
1.Machine Learning:
Uses an XGBoost model for predictions.
Encodes categorical data and balances classes using SMOTE.
2.Frontend:
Provides a user-friendly interface to input loan details and view predictions.
3.Backend:
Built with Flask, handles data preprocessing and API integration.

Requirements
Ensure the following are installed on your system:
1.Python 3.12.4 (or compatible version)
2.Flask
3.XGBoost
4.Pandas
5.Scikit-learn
6.Imbalanced-learn
7.HTML/CSS for frontend design

To install dependencies, use:
pip install -r requirements.txt

How to Run
1.Clone or Extract the Project:
  Extract the Application.zip file.
2.Setup the Environment:
  Navigate to the project folder:
  cd application/
  Install the required Python packages:
  pip install -r requirements.txt
3.Start the Application:
  Run the Flask application:
  python app.py
  The terminal will display a server link similar to:
  * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
4.Access the Frontend:
  Click the server link shown in the terminal or open it in your browser.
  Enter the required loan details and submit the form to view the prediction results.

Testing the Application
1.Run the Flask Application:
  Execute app.py and open the server link displayed in the terminal.
2.Interact with the Frontend:
  Enter valid loan details on the provided input form.
  Submit the data to receive a prediction about the default risk.
3.Error Handling:
 If inputs are invalid or missing, an error message will be displayed.  

Troubleshooting
1.Flask Errors:
  Ensure all dependencies are installed.
  Check for port conflicts and use flask run --host=0.0.0.0 --port=<PORT> if needed.
2.Model Errors:
  Verify the dataset is correctly placed and formatted.
  Retrain the model if the pre-trained .pkl files are corrupted or missing.


  

