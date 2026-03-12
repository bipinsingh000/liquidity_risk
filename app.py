import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("liquidity_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💰 Liquidity Risk Prediction System")

st.write("Enter company financial data to predict financial risk.")

st.subheader("Enter Financial Values")

# Input fields
debt_equity = st.text_input("Debt Equity Ratio (0 - 10)", "1.0")
total_liabilities = st.text_input("Total Liabilities (0 - 1,000,000,000)", "1000000")
return_assets = st.text_input("Return on Assets (-1 to 1)", "0.05")
current_assets = st.text_input("Current Assets (0 - 1,000,000,000)", "1000000")
total_assets = st.text_input("Total Assets (0 - 2,000,000,000)", "1000000")
net_income = st.text_input("Net Income (-1,000,000,000 to 1,000,000,000)", "100000")


if st.button("Predict Financial Risk"):

    try:
        # convert input
        debt_equity = float(debt_equity)
        total_liabilities = float(total_liabilities)
        return_assets = float(return_assets)
        current_assets = float(current_assets)
        total_assets = float(total_assets)
        net_income = float(net_income)

        # --------------------------
        # Limit Validation
        # --------------------------

        if not (0 <= debt_equity <= 10):
            st.error("Debt Equity Ratio must be between 0 and 10")
            st.stop()

        if not (0 <= total_liabilities <= 1_000_000_000):
            st.error("Total Liabilities must be between 0 and 1,000,000,000")
            st.stop()

        if not (-1 <= return_assets <= 1):
            st.error("Return on Assets must be between -1 and 1")
            st.stop()

        if not (0 <= current_assets <= 1_000_000_000):
            st.error("Current Assets must be between 0 and 1,000,000,000")
            st.stop()

        if not (0 <= total_assets <= 2_000_000_000):
            st.error("Total Assets must be between 0 and 2,000,000,000")
            st.stop()

        if not (-1_000_000_000 <= net_income <= 1_000_000_000):
            st.error("Net Income must be between -1,000,000,000 and 1,000,000,000")
            st.stop()

        # --------------------------
        # Log transform
        # --------------------------

        total_liabilities = np.log1p(total_liabilities)
        current_assets = np.log1p(current_assets)
        total_assets = np.log1p(total_assets)
        net_income = np.log1p(net_income)

        # input array
        input_data = np.array([[

            debt_equity,
            total_liabilities,
            return_assets,
            current_assets,
            total_assets,
            net_income

        ]])

        # scaling
        input_scaled = scaler.transform(input_data)

        # prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"⚠️ High Financial Risk\n\nProbability: {probability:.2f}")
        else:
            st.success(f"✅ Low Financial Risk\n\nProbability: {probability:.2f}")

    except:
        st.warning("Please enter valid numeric values only.")