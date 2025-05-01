import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="GameStop (GME) Stock Tracker")
st.title("📈 GameStop (GME) Stock - Last 30 Days")

# ✅ Add caching function
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    return yf.download("GME", period="30d")

# ✅ Call the cached function
df = load_data()

# Display table
st.subheader("Historical Price Table (Last 30 Days)")
st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']])

# Display chart
st.subheader("Closing Price Chart")
st.line_chart(df['Close'])

# CSV download button
st.download_button(
    label="Download data as CSV",
    data=df.to_csv().encode('utf-8'),
    file_name='gme_last_30_days.csv',
    mime='text/csv',
)
