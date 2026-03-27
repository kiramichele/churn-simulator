import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

@st.cache_data
def load_and_train():
    df = pd.read_csv('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df = df.drop('customerID', axis=1)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

model, scaler = load_and_train()

st.title("📊 Customer Churn Predictor")
st.markdown("Enter customer details to predict churn probability.")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
    total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges))
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

with col2:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map = {"Bank transfer": 0, "Credit card": 1, "Electronic check": 2, "Mailed check": 3}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
binary_map = {"No": 0, "Yes": 1, "No internet service": 0}

input_data = pd.DataFrame([[
    1, 0, 1, 1, tenure, 1, 1,
    internet_map[internet], binary_map[online_security], 1,
    1, binary_map[tech_support], 1, 1,
    contract_map[contract], binary_map[paperless],
    payment_map[payment], monthly_charges, total_charges
]], columns=['gender','SeniorCitizen','Partner','Dependents','tenure',
             'PhoneService','MultipleLines','InternetService','OnlineSecurity',
             'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
             'StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
             'MonthlyCharges','TotalCharges'])

if st.button("Predict Churn Risk"):
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    
    st.markdown("---")
    if prob > 0.6:
        st.error(f"⚠️ High Churn Risk: {prob:.1%}")
        st.markdown("**Recommendation:** Offer contract upgrade incentive immediately.")
    elif prob > 0.3:
        st.warning(f"⚡ Medium Churn Risk: {prob:.1%}")
        st.markdown("**Recommendation:** Monitor and offer loyalty discount.")
    else:
        st.success(f"✅ Low Churn Risk: {prob:.1%}")
        st.markdown("**Recommendation:** Customer is stable. No action needed.")
