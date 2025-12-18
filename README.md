# Telecom Customer Churn Prediction

End-to-end machine learning project to predict customer churn for a telecom company using the Telco Customer Churn dataset. The notebook covers EDA, feature engineering, model training and a prediction demo.
## 1. Problem Statement

Telecom companies lose revenue when existing customers cancel their services.  
This project builds a classification model that predicts whether a customer will churn in the near future and highlights the key factors behind churn.[file:2]
## 2. Dataset

- Dataset: Telco Customer Churn (7,032 customers, 21 features, target column `Churn`).[file:2]
- Features include demographics, contract details, internet/phone services and monthly/total charges.[file:2]

The raw CSV is placed under `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
## 3. Approach

1. **Data cleaning**
   - Fixed `TotalCharges` type (object → float) and removed 11 rows with missing values.[file:2]
   - Dropped `customerID` as a non-informative identifier.

2. **Exploratory Data Analysis**
   - Visualised churn distribution and class imbalance.
   - Analysed tenure, monthly charges and total charges distributions.
   - Explored churn rate by contract type and other categorical features.[file:2]

3. **Feature Engineering**
   - Encoded `Churn` as 0/1.
   - One-hot encoded all categorical variables with `pd.get_dummies(..., drop_first=True)`.
   - Scaled numerical features (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler`.[file:2]

4. **Modelling**
   - Train–test split (80/20, stratified on `Churn`).[file:2]
   - Trained **Logistic Regression** and **XGBoost** models.
   - Evaluated using Accuracy, Precision, Recall, F1-score and ROC-AUC plus confusion matrix and ROC curve.[file:2]
## 4. Results

| Model               | Accuracy | Precision | Recall | F1   | ROC-AUC |
|---------------------|----------|-----------|--------|------|---------|
| Logistic Regression | 0.80–0.81| ~0.65     | ~0.57  | ~0.61| ~0.83–0.84 |
| XGBoost             | ~0.78    | ~0.61     | ~0.52  | ~0.56| ~0.83     |[file:2]

- Logistic Regression performed slightly better and is used as the final model.
- High churn risk is associated with month-to-month contracts, short tenure, high monthly charges and lack of security/support add-ons.[file:2]
## 5. Prediction Demo

The notebook defines a helper function:


You can pass raw customer details (gender, tenure, contract type, services, charges) and receive:

- `proba`: predicted probability of churn  
- `label`: `"Churn"` or `"No Churn"`

Several sample customer profiles are included to demonstrate high-risk and low-risk predictions.[file:2]
