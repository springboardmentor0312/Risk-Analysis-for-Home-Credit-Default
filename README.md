# Home Credit Default Risk Analysis Application

This application predicts the likelihood of loan default using an XGBoost machine learning model. It includes a Flask backend, a web-based frontend, and a robust pipeline for preprocessing and modeling.

Application ├── app/ │ ├── static/ # Static files (CSS, JS, Images) │ ├── templates/ # HTML files (Frontend) │ ├── model/ # Trained models and encoders (xgb_model.pkl, label_encoders.pkl) │ ├── app.py # Main Flask application │ ├── preprocess.py # Data preprocessing utilities │ └── requirements.txt # Python dependencies ├── dataset/ │ └── HCD(Preprocessed).csv # Preprocessed dataset └── README.md # Project documentation

