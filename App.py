import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# --- Load Model and Scaler ---
@st.cache_resource
def load_model_and_scaler():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "rf.pkl")
    scaler_path = os.path.join(current_dir, "scaler.pkl")

    # Load model
    loaded_model = joblib.load(model_path)

    # Load scaler if available
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        fitted = True
    else:
        scaler = StandardScaler()
        fitted = False

    return loaded_model, scaler, fitted

# Load model/scaler once
loaded_model, scaler, fitted = load_model_and_scaler()

# --- Streamlit UI ---
st.set_page_config(page_title="Bank Customer Churn App", page_icon="🏦", layout="centered")

st.title("🏦 Bank Customer Churn App")
st.markdown("""
This app predicts whether a **bank customer is likely to churn**
based on demographic and account features.
""")

# --- Input Fields ---
customer_id = st.text_input("Customer ID (optional)", value="12345")
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
country = st.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10, value=5)
balance = st.number_input("Account Balance", min_value=0.0, value=60000.0, step=1000.0)
products_number = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
credit_card = st.selectbox("Has Credit Card?", ["No", "Yes"])
active_member = st.selectbox("Is Active Member?", ["No", "Yes"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

# --- Mappings ---
country_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}
gender_mapping = {'Male': 0, 'Female': 1}
credit_card_mapping = {'No': 0, 'Yes': 1}
active_mapping = {'No': 0, 'Yes': 1}

# --- Prepare Input ---
country_val = country_mapping[country]
gender_val = gender_mapping[gender]
credit_card_val = credit_card_mapping[credit_card]
active_val = active_mapping[active_member]

input_data = pd.DataFrame([[
    credit_score, country_val, gender_val, age, tenure, balance,
    products_number, credit_card_val, active_val, estimated_salary
]], columns=[
    'credit_score', 'country', 'gender', 'age', 'tenure', 'balance',
    'products_number', 'credit_card', 'active_member', 'estimated_salary'
])

# --- Predict Button ---
if st.button("🔍 Predict Churn"):
    try:
        # If scaler is not fitted, fit it on the input (fallback)
        if not fitted:
            scaler.fit(input_data)

        scaled_input = scaler.transform(input_data)

        prediction = loaded_model.predict(scaled_input)[0]
        prediction_prob = loaded_model.predict_proba(scaled_input)[0][1]

        st.subheader("🧾 Classification Result")

        if prediction == 1:
            st.error(f"⚠️ This customer is **likely to churn**.\n\n**Churn Probability:** {prediction_prob:.2f}")
        else:
            st.success(f"✅ This customer is **likely to stay**.\n\n**Churn Probability:** {prediction_prob:.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption("Model: Random Forest Classifier | Encoding: LabelEncoder + StandardScaler")
