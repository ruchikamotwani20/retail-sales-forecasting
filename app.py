import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

# Load the model and scaler
model = load_model('sales_lstm_model.h5', compile=False)
scaler = joblib.load('scaler_4features.save')  # ✅ Trained on 2020–2025 data

st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")

# Title
st.title("📊 AI-Powered Retail Sales Forecast Dashboard")
st.markdown("Predict monthly U.S. retail sales using LSTM + Economic Indicators")

# Sidebar Input
st.sidebar.header("📥 Upload Input CSV")
uploaded_file = st.sidebar.file_uploader("Upload last 12 months of data as CSV", type=["csv"])
st.sidebar.caption("CSV must contain 12 rows and 4 columns: Sales_Amount, CPI, Unemployment, Month")

# Forecast Month Selection
selected_date = st.sidebar.date_input("📅 Select Forecast Month", datetime.today())

# Prediction
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ['Sales_Amount', 'CPI', 'Unemployment', 'Month']
        if not all(col in df.columns for col in required_cols):
            st.error("❌ CSV must contain: Sales_Amount, CPI, Unemployment, Month")
        elif len(df) != 12:
            st.error("❌ CSV must contain exactly 12 rows")
        else:
            X_input = df[required_cols].values
            scaled_input = scaler.transform(X_input)
            X_input_reshaped = scaled_input.reshape(1, 12, 4)

            pred_scaled = model.predict(X_input_reshaped)
            padded = np.zeros((1, 4))
            padded[0, 0] = pred_scaled[0, 0]
            predicted_sales = scaler.inverse_transform(padded)[0, 0]

            st.success(f"📈 Predicted Sales for {selected_date.strftime('%B %Y')}: **${predicted_sales:,.2f}**")

            # Plot
            past_months = pd.date_range(end=selected_date - pd.DateOffset(months=1), periods=12, freq='MS')
            chart_data = pd.DataFrame({
                "Month": list(past_months.strftime('%b %Y')) + [selected_date.strftime('%b %Y')],
                "Sales": list(df['Sales_Amount'].values) + [predicted_sales]
            })

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(chart_data["Month"], chart_data["Sales"], marker='o')
            ax.set_title("📉 Last 12 Months + Predicted Sales")
            ax.set_ylabel("Sales ($)")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

            # Download CSV
            output = pd.DataFrame({"Forecast_Month": [selected_date.strftime('%Y-%m')], "Predicted_Sales": [predicted_sales]})
            csv = output.to_csv(index=False)
            st.download_button("📥 Download Prediction CSV", data=csv, file_name="forecast.csv")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("👈 Please upload a valid CSV file to begin forecasting.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, LSTM & Economic Data (2020–2025 adjusted)")