import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the fitted model
model = joblib.load('sarima_model.pkl')

st.title("Yahoo Stock SARIMA Forecast")

# User input: number of days to forecast
n_days = st.number_input("Days to forecast", min_value=1, max_value=30, value=7)

if st.button("Forecast"):
    # Now n_days is defined, so you can use it here
    forecast = model.forecast(steps=n_days)
    st.write("Forecasted values:")
    st.write(forecast)

    # Plot the forecast
    fig, ax = plt.subplots()
    pd.Series(forecast).plot(ax=ax, label="Forecast")
    ax.set_title("SARIMA Forecast")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)