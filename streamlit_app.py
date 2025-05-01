import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Stock Tracker", layout="wide")

# ----------- Input for Dynamic Ticker Name ---------
ticker_input = st.text_input("Enter Ticker Symbol", "GME").upper()

# ----------- Load Data with Caching -----------
@st.cache_data(ttl=3600)
def load_price_data(ticker):
    df = yf.download(ticker, period="3y")  # Changed to last 3 years
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

@st.cache_data(ttl=3600)
def load_fundamentals(ticker):
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    try:
        return {
            "EPS (TTM)": info.get("trailingEps", "N/A"),
            "Revenue (TTM)": f"${info.get('totalRevenue', 0):,}" if info.get("totalRevenue") else "N/A"
        }
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def load_eps_history(ticker):
    ticker_obj = yf.Ticker(ticker)
    try:
        q_eps = ticker_obj.quarterly_earnings
        y_eps = ticker_obj.earnings
    except Exception:
        q_eps = pd.DataFrame()  # Ensure it's a DataFrame even when there's an error
        y_eps = pd.DataFrame()  # Ensure it's a DataFrame even when there's an error
    return q_eps, y_eps

# ----------- Calculate Indicators -----------
def add_analytics(df):
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_25'] = df['Close'].rolling(window=25).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    df_reset = df.reset_index()
    df_reset['Date_ordinal'] = df_reset['Date'].map(pd.Timestamp.toordinal)
    coeffs = np.polyfit(df_reset['Date_ordinal'], df_reset['Close'], 1)
    df['Trend'] = coeffs[0] * df.index.map(pd.Timestamp.toordinal) + coeffs[1]
    return df

# ----------- Load Data -----------
df = load_price_data(ticker_input)
df = add_analytics(df)
last_30 = df.tail(30)
financials = load_fundamentals(ticker_input)
q_eps, y_eps = load_eps_history(ticker_input)

# ----------- Price Table -----------
st.subheader(f"📅 Historical Price Table (Last 30 Days) - {ticker_input}")
st.dataframe(last_30[['Open', 'High', 'Low', 'Close', 'Volume']])

# ----------- Altair Chart: Price + Trend + MAs -----------
st.subheader(f"📈 Price Chart with Trend & MAs - {ticker_input}")

price_chart_data = last_30.reset_index()

base = alt.Chart(price_chart_data).encode(
    x='Date:T'
)

price_line = base.mark_line(color='white', strokeWidth=3).encode(
    y=alt.Y('Close:Q', scale=alt.Scale(domain=[0, price_chart_data['Close'].max() * 1.1]), title='Price')
)

ma_5 = base.mark_line(color='blue', strokeDash=[5, 3], size=3).encode(
    y='MA_5:Q',
    tooltip=[alt.Tooltip('Date:T', title='Date'),
             alt.Tooltip('Close:Q', title='Price'),
             alt.Tooltip('MA_5:Q', title='5-day MA')]
)

ma_25 = base.mark_line(color='green', strokeDash=[5, 3], size=3).encode(
    y='MA_25:Q',
    tooltip=[alt.Tooltip('Date:T', title='Date'),
             alt.Tooltip('Close:Q', title='Price'),
             alt.Tooltip('MA_25:Q', title='25-day MA')]
)

ma_200 = base.mark_line(color='red', strokeDash=[5, 3], size=3).encode(
    y='MA_200:Q',
    tooltip=[alt.Tooltip('Date:T', title='Date'),
             alt.Tooltip('Close:Q', title='Price'),
             alt.Tooltip('MA_200:Q', title='200-day MA')]
)

trend = base.mark_line(color='#FF9933', opacity=0.5).encode(
    y='Trend:Q',
    tooltip=[alt.Tooltip('Date:T', title='Date'),
             alt.Tooltip('Close:Q', title='Price'),
             alt.Tooltip('Trend:Q', title='Trend')]
)

# Combining all into one chart
st.altair_chart((price_line + ma_5 + ma_25 + ma_200 + trend).properties(height=400), use_container_width=True)

# ----------- Average Volume Chart ----------- 
st.subheader(f"📊 Daily Volume (Last 30 Days) - {ticker_input}")

avg_volume = last_30['Volume'].mean()

# Volume chart styled like price chart
volume_chart = alt.Chart(price_chart_data).mark_bar(color="#4A90E2", opacity=0.6).encode(
    x='Date:T',
    y=alt.Y('Volume:Q', title='Volume'),
    tooltip=[alt.Tooltip('Date:T', title='Date'),
             alt.Tooltip('Volume:Q', title='Volume')]
) + alt.Chart(price_chart_data).mark_rule(color="red", strokeDash=[4,2]).encode(
    y=alt.value(avg_volume)
)

st.altair_chart(volume_chart.properties(height=200), use_container_width=True)
st.caption(f"🔻 Average Volume: {int(avg_volume):,}")

# ----------- Financials Summary -----------
st.subheader(f"💵 Key Financial Metrics - {ticker_input}")
for k, v in financials.items():
    st.markdown(f"**{k}:** {v}")

# ----------- EPS Display -----------
st.subheader(f"🧾 Earnings Per Share (EPS) - {ticker_input}")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Last 8 Quarters EPS:**")
    if isinstance(q_eps, pd.DataFrame) and not q_eps.empty:
        st.table(q_eps.head(8)[['Earnings']])
    else:
        st.warning("Quarterly EPS data unavailable. Please check if the data exists on Yahoo Finance.")

with col2:
    st.markdown("**Annual EPS (Last 4 Years):**")
    if isinstance(y_eps, pd.DataFrame) and not y_eps.empty:
        st.table(y_eps.tail(4)[['Earnings']])
    else:
        st.warning("Annual EPS data unavailable. Please check if the data exists on Yahoo Finance.")

# ----------- CSV Download Button -----------
st.download_button(
    label="⬇️ Download full dataset as CSV",
    data=df.to_csv().encode('utf-8'),
    file_name=f'{ticker_input}_stock_data.csv',
    mime='text/csv',
)
