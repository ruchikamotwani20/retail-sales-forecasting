**Project Title:**

Multivariate Time Series Forecasting of Retail Sales Using Economic Indicators

**Author:** 

Ruchika Motwani

**Project Duration:** 

Feb 2025 – May 2025

**Overview:**

This project addresses the challenge of accurately forecasting monthly retail sales by incorporating macroeconomic variables. While traditional sales forecasting methods rely only on historical sales data, this project leverages additional context from external indicators like the Consumer Price Index (CPI) and Unemployment Rate to improve model precision.

The project involves the implementation of both traditional machine learning models and advanced deep learning techniques (LSTM) and compares their performance using standard evaluation metrics.

**Objective:**

Forecast monthly U.S. retail sales using multivariate time series data

Understand the impact of CPI and Unemployment on retail trends

Evaluate and compare model performance: Linear Regression, Random Forest, XGBoost, and LSTM

Build an interactive dashboard for scenario-based forecasting

**Data Sources:**

U.S. Retail Sales Data (2020–2025)
Source: U.S. Census Bureau – Retail and Food Services Sales
Link: https://www.census.gov/retail/index.html

Consumer Price Index (CPI)
Source: Federal Reserve Economic Data (FRED) – CPI for All Urban Consumers
Link: https://fred.stlouisfed.org/series/CPIAUCNS

Unemployment Rate
Source: Federal Reserve Economic Data (FRED) – Unemployment Rate
Link: https://fred.stlouisfed.org/series/UNRATE

**Features Used for Modeling:**

| Feature        | Description                                    |
| -------------- | ---------------------------------------------- |
| `Sales_Amount` | Monthly retail sales in USD (Target variable)  |
| `CPI`          | Consumer Price Index, measures inflation       |
| `Unemployment` | Unemployment rate (%)                          |
| `Month`        | Numeric month (1 to 12) to capture seasonality |

**Modeling Approach:**

**Traditional Models**

  Linear Regression

  Random Forest Regressor

  XGBoost Regressor

**Deep Learning**

  Long Short-Term Memory (LSTM) neural network

  Sequence window of 12 months to predict the next month

  Feature scaling using MinMaxScaler

  Dropout used to prevent overfitting

  Early stopping used during training
  
**Evaluation Metrics:**

RMSE (Root Mean Squared Error): Measures average prediction error

R² Score (Coefficient of Determination): Indicates variance explained by the model

**Performance Summary:**

| Model                    | RMSE         | R² Score |
| ------------------------ | ------------ | -------- |
| Linear Regression        | 19,514.08    | -2.13    |
| Random Forest            | 22,506.45    | -3.17    |
| XGBoost                  | 21,443.19    | -2.78    |
| **LSTM (Deep Learning)** | **4,668.68** | **0.78** |

**Exploratory Analysis Highlights:**

Seasonality: Spikes in sales around November–December

CPI Correlation: Strong positive correlation with sales

Unemployment Correlation: Strong negative correlation with sales

COVID Impact: Drop in early 2020 and recovery trend post mid-2020

**Streamlit App – Forecasting Dashboard:**

A real-time forecasting tool was built using Streamlit to enable interactive use of the LSTM model.

Dashboard Features:

Upload or manually enter past 12 months of data

Adjust CPI and Unemployment Rate

Choose month to predict

Visualize forecast alongside previous sales

Download results as CSV
<img width="1081" alt="Screenshot 2025-05-25 at 1 45 53 PM" src="https://github.com/user-attachments/assets/e43b002d-4163-479f-a038-08eb8b81d68a" />

**Future Enhancements:**

Add variables like interest rate, fuel price, or consumer sentiment

Explore GRUs, BiLSTMs, and Transformer architectures

Enable multi-month forecasting

Integrate with live APIs for real-time prediction

