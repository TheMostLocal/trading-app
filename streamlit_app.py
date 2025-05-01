import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Set page title
st.set_page_config(page_title="GameStop (GME) Stock Tracker")

# Header
st.title("📈 GameStop (GME) Stock - Last 30 Days")

# Download GME data from Yahoo Finance
ticker = "GME"
gme = yf.Ticker(ticker)
hist = gme.history(period="30d")

# Display data as table
st.subheader("Historical Price Table (Last 30 Days)")
st.dataframe(hist[['Open', 'High', 'Low', 'Close', 'Volume']])

# Plot historical close prices
st.subheader("Closing Price Chart")
st.line_chart(hist['Close'])

# Optional: add raw data download
st.download_button(
    label="Download data as CSV",
    data=hist.to_csv().encode('utf-8'),
    file_name='gme_last_30_days.csv',
    mime='text/csv',
)
