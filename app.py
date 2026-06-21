import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("log_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")

st.title("🏦 Loan Approval Prediction System")

# Numerical inputs
Applicant_Income = st.number_input("Applicant Income", min_value=0.0)
Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0.0)
Age = st.number_input("Age", min_value=18)
Dependents = st.number_input("Dependents", min_value=0)
Existing_Loans = st.number_input("Existing Loans", min_value=0)
Savings = st.number_input("Savings", min_value=0.0)
Collateral_Value = st.number_input("Collateral Value", min_value=0.0)
Loan_Amount = st.number_input("Loan Amount", min_value=0.0)
Loan_Term = st.number_input("Loan Term", min_value=0)
Education_Level = st.number_input("Education Level", min_value=0)

# One-hot encoded columns (0 or 1)
Employment_Status_Salaried = st.selectbox(
    "Employment_Status_Salaried", [0, 1]
)

Employment_Status_Self_employed = st.selectbox(
    "Employment_Status_Self-employed", [0, 1]
)

Employment_Status_Unemployed = st.selectbox(
    "Employment_Status_Unemployed", [0, 1]
)

Marital_Status_Single = st.selectbox(
    "Marital_Status_Single", [0, 1]
)

Loan_Purpose_Car = st.selectbox(
    "Loan_Purpose_Car", [0, 1]
)

Loan_Purpose_Education = st.selectbox(
    "Loan_Purpose_Education", [0, 1]
)

Loan_Purpose_Home = st.selectbox(
    "Loan_Purpose_Home", [0, 1]
)

Loan_Purpose_Personal = st.selectbox(
    "Loan_Purpose_Personal", [0, 1]
)

Property_Area_Semiurban = st.selectbox(
    "Property_Area_Semiurban", [0, 1]
)

Property_Area_Urban = st.selectbox(
    "Property_Area_Urban", [0, 1]
)

Gender_Male = st.selectbox(
    "Gender_Male", [0, 1]
)

Employer_Category_Government = st.selectbox(
    "Employer_Category_Government", [0, 1]
)

Employer_Category_MNC = st.selectbox(
    "Employer_Category_MNC", [0, 1]
)

Employer_Category_Private = st.selectbox(
    "Employer_Category_Private", [0, 1]
)

Employer_Category_Unemployed = st.selectbox(
    "Employer_Category_Unemployed", [0, 1]
)

# Required to generate squared features
DTI_Ratio = st.number_input("DTI Ratio", min_value=0.0)
Credit_Score = st.number_input("Credit Score", min_value=0.0)

# Engineered features
DTI_Ratio_sq = DTI_Ratio ** 2
Credit_Score_sq = Credit_Score ** 2

if st.button("Predict"):

    features = np.array([[
        Applicant_Income,
        Coapplicant_Income,
        Age,
        Dependents,
        Existing_Loans,
        Savings,
        Collateral_Value,
        Loan_Amount,
        Loan_Term,
        Education_Level,
        Employment_Status_Salaried,
        Employment_Status_Self_employed,
        Employment_Status_Unemployed,
        Marital_Status_Single,
        Loan_Purpose_Car,
        Loan_Purpose_Education,
        Loan_Purpose_Home,
        Loan_Purpose_Personal,
        Property_Area_Semiurban,
        Property_Area_Urban,
        Gender_Male,
        Employer_Category_Government,
        Employer_Category_MNC,
        Employer_Category_Private,
        Employer_Category_Unemployed,
        DTI_Ratio_sq,
        Credit_Score_sq
    ]])

    # Scale input
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)

    # Probability
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"Approval Probability: {probability:.2%}")