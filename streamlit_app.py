import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="GameStop (GME) Stock Tracker", layout="wide")
st.title("📊 GameStop (GME) Stock - Last 30 Days")

# ----------- Load Data with Caching -----------
@st.cache_data(ttl=3600)
def load_price_data():
    df = yf.download("GME", period="200d")
    
    # Flatten MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    return df

@st.cache_data(ttl=3600)
def load_fundamentals():
    ticker = yf.Ticker("GME")
    info = ticker.info
    bs = ticker.balance_sheet

    try:
        dta_ratio = round(bs.loc["Total Liab"][0] / bs.loc["Total Assets"][0], 2)
    except Exception:
        dta_ratio = "N/A"

    financials = {
        "EPS (TTM)": info.get("trailingEps", "N/A"),
        "Revenue (TTM)": f"${info.get('totalRevenue', 0):,}" if info.get("totalRevenue") else "N/A",
        "Debt-to-Assets Ratio": dta_ratio
    }
    return financials

# ----------- Calculate MAs and Trend Line -----------
def add_analytics(df):
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_25'] = df['Close'].rolling(window=25).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    df_reset = df.reset_index()
    df_reset['Date_ordinal'] = df_reset['Date'].map(pd.Timestamp.toordinal)
    coeffs = np.polyfit(df_reset['Date_ordinal'], df_reset['Close'], 1)
    df['Trend'] = coeffs[0] * df.index.map(pd.Timestamp.toordinal) + coeffs[1]
    
    return df

# ----------- Load and Process Data -----------
df = load_price_data()
if df.empty:
    st.error("⚠️ No price data returned. You may have hit the Yahoo Finance rate limit.")
    st.stop()

df = add_analytics(df)
last_30 = df.tail(30)
financials = load_fundamentals()

# ----------- Price Table -----------
st.subheader("📅 Historical Price Table (Last 30 Days)")
st.dataframe(last_30[['Open', 'High', 'Low', 'Close', 'Volume']])

# ----------- Chart with Trend & MA -----------
st.subheader("📈 Closing Price with Trend Line & Moving Averages (Last 30 Days)")
st.line_chart(last_30[['Close', 'MA_5', 'MA_25', 'Trend']])

# ----------- Financial Metrics -----------
st.subheader("💵 Key Financial Metrics")
for k, v in financials.items():
    st.markdown(f"**{k}:** {v}")

# ----------- CSV Download Button -----------
st.download_button(
    label="⬇️ Download full dataset as CSV",
    data=df.to_csv().encode('utf-8'),
    file_name='gme_stock_data.csv',
    mime='text/csv',
)
