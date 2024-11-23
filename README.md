# Home Credit Default Prediction

## Overview

This project focuses on predicting the likelihood of default on home loans using machine learning techniques. By analyzing real-world borrower data, we aim to help financial institutions reduce risk and improve their lending decisions.

## Project Goals

1. **Data Exploration**: Understand the dataset and its characteristics through exploratory data analysis (EDA).
2. **Feature Identification**: Identify key factors that influence loan defaults using data visualization and statistical methods.
3. **Model Development**: Create predictive models using various machine learning algorithms to assess the risk of default.
4. **Performance Evaluation**: Measure model effectiveness using metrics such as accuracy, precision, recall, and F1 score.
5. **Insights and Recommendations**: Provide actionable insights to help lenders mitigate credit risk.

## Dataset

The project utilizes a dataset provided by Home Credit, which includes various CSV files containing information about loan applicants, their financial profiles, credit histories, and loan terms. Key files include:

- `application_train.csv`: Training data with target labels (default or not).
- `application_test.csv`: Test data without target labels.
- Additional files containing credit card balances, previous applications, and more.

## Methodology

1. **Exploratory Data Analysis (EDA)**: 
   - Analyzed the distribution of loan defaults and other borrower characteristics.
   - Visualized relationships between features and the target variable.

2. **Data Preprocessing**:
   - Handled missing values and outliers.
   - Normalized numerical features and encoded categorical variables for machine learning models.

3. **Modeling**:
   - Implemented various algorithms, including Logistic Regression, Random Forest, and XGBoost.
   - Selected CatBoost, the best-performing model based on evaluation metrics.

4. **Deployment**:
   - Developed a web application using Flask and Deployed using Render to allow users to input borrower data and receive predictions on loan defaults.

## Results

The project successfully predicts loan defaults with a focus on accuracy and interpretability. Key findings include:

- Younger borrowers tend to have a higher default rate.
- Certain income levels and employment statuses are associated with increased risk.
- The best-performing model was selected based on comprehensive evaluation metrics.

## Future Work

- Enhance the model with live data integration for real-time predictions.
- Explore advanced techniques like deep learning for improved accuracy.
- Develop user-friendly dashboards for stakeholders to visualize predictions and insights.

## Getting Started

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/springboardmentor0312/Risk-Analysis-for-Home-Credit-Default/tree/group-1.git
   cd Risk-Analysis-for-Home-Credit-Default
2. **CInstall required packages**:
   ```bash
   pip install -r requirements.txt
3. **Run the Flask application**:
   ```bash
   python app.py
4. **Access the application**:
   -Open your web browser and go to https://house-credit-default-infosys-6.onrender.com/

## Team Members

- **S. Chaitanya Deepthi**: EDA and data cleaning.
- **Sk. Shakila**: Model training and optimization.
- **Smruti Deshpande**: Backend development and deployment.
- **C. Sahi**: Frontend development and user interface.
