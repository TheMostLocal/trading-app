import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Stock Tracker", layout="wide")
st.title("📊 Stock Tracker Dashboard")

# ----------- Load Data with Caching -----------
@st.cache_data(ttl=3600)
def load_price_data(ticker):
    df = yf.download(ticker, period="3y")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

@st.cache_data(ttl=3600)
def load_fundamentals(ticker):
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    bs = ticker_obj.balance_sheet
    try:
        dta_ratio = round(bs.loc["Total Liab"][0] / bs.loc["Total Assets"][0], 2)
    except Exception:
        dta_ratio = "N/A"
    return {
        "EPS (TTM)": info.get("trailingEps", "N/A"),
        "Revenue (TTM)": f"${info.get('totalRevenue', 0):,}" if info.get("totalRevenue") else "N/A",
        "Debt-to-Assets Ratio": dta_ratio,
        "PE Ratio": info.get("trailingPE", "N/A"),
        "Market Cap": f"${info.get('marketCap', 0):,}" if info.get("marketCap") else "N/A",
    }

@st.cache_data(ttl=3600)
def load_eps_history(ticker):
    ticker_obj = yf.Ticker(ticker)
    try:
        q_eps = ticker_obj.quarterly_earnings
        y_eps = ticker_obj.earnings
    except Exception:
        q_eps = pd.DataFrame()
        y_eps = pd.DataFrame()
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

# ----------- Get Stock Data -----------
ticker = st.text_input("Enter Stock Ticker (e.g., GME, AAPL, AMZN):", "GME")
df = load_price_data(ticker)
df = add_analytics(df)

# ----------- Price Data for Last 30 Days ----------- 
last_30 = df.tail(30)
financials = load_fundamentals(ticker)
q_eps, y_eps = load_eps_history(ticker)

# ----------- Price Table ----------- 
st.subheader("📅 Historical Price Table (Last 30 Days)")
st.dataframe(last_30[['Open', 'High', 'Low', 'Close', 'Volume']])

# ----------- Price Chart with Trend & MAs ----------- 
st.subheader("📈 Price Chart with Trend & MAs")

price_chart_data = last_30.reset_index()

base = alt.Chart(price_chart_data).encode(
    x='Date:T'
)

price_line = base.mark_line(color='white', strokeWidth=3).encode(
    y=alt.Y('Close:Q', scale=alt.Scale(domain=[0, price_chart_data['Close'].max() * 1.1]), title='Price')
)

ma_5 = base.mark_line(color='blue', strokeDash=[5, 3]).encode(y='MA_5:Q', tooltip=['Date', 'Close', 'MA_5'])
ma_25 = base.mark_line(color='green', strokeDash=[5, 3]).encode(y='MA_25:Q', tooltip=['Date', 'Close', 'MA_25'])
ma_200 = base.mark_line(color='red', strokeDash=[5, 3]).encode(y='MA_200:Q', tooltip=['Date', 'Close', 'MA_200'])
trend = base.mark_line(color='#FF9933', opacity=0.5).encode(y='Trend:Q')

st.altair_chart((price_line + ma_5 + ma_25 + ma_200 + trend).properties(height=400), use_container_width=True)

# ----------- Sentiment (Buy/Hold/Sell) ----------- 
st.subheader("📉 Sentiment")
# You can include logic to define the sentiment based on the latest price and MAs
latest_close = last_30['Close'].iloc[-1]
ma_5_latest = last_30['MA_5'].iloc[-1]
ma_25_latest = last_30['MA_25'].iloc[-1]
ma_200_latest = last_30['MA_200'].iloc[-1]

if latest_close > ma_5_latest and ma_5_latest > ma_25_latest and ma_25_latest > ma_200_latest:
    sentiment = "Buy"
elif latest_close < ma_5_latest and ma_5_latest < ma_25_latest and ma_25_latest < ma_200_latest:
    sentiment = "Sell"
else:
    sentiment = "Hold"

st.markdown(f"**Sentiment: {sentiment}**")

# ----------- Average Volume Chart ----------- 
st.subheader("📊 Average Daily Volume")
avg_volume = last_30['Volume'].mean()
st.caption(f"🔻 Average Volume: {int(avg_volume):,}")

volume_chart = alt.Chart(price_chart_data).mark_bar(color="#4A90E2").encode(
    x='Date:T',
    y='Volume:Q'
) + alt.Chart(price_chart_data).mark_rule(color="red", strokeDash=[4,2]).encode(
    y=alt.value(avg_volume)
)

st.altair_chart(volume_chart.properties(height=200), use_container_width=True)

# ----------- Financial Metrics ----------- 
st.subheader("💵 Key Financial Metrics")
# Ensure all values that can be converted to numbers are converted
# Handle the case where values might be 'N/A' or string-based
cleaned_financials = {}
for key, value in financials.items():
    try:
        # Try to convert to float, if not possible, keep as is (for strings like "N/A")
        cleaned_financials[key] = float(value) if isinstance(value, (int, float)) else value
    except ValueError:
        cleaned_financials[key] = value

# Convert to DataFrame and sort values
financial_metrics = pd.DataFrame.from_dict(cleaned_financials, orient='index', columns=["Value"])

# Convert any strings to NaN for sorting and handle
financial_metrics['Value'] = pd.to_numeric(financial_metrics['Value'], errors='coerce')

# Sort by value, descending
financial_metrics_sorted = financial_metrics.sort_values(by="Value", ascending=False)

st.table(financial_metrics_sorted)

# ----------- EPS Display ----------- 
st.subheader("🧾 Earnings Per Share (EPS)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Last 8 Quarters EPS:**")
    if not q_eps.empty:
        st.table(q_eps.head(8)[['Earnings']])
    else:
        st.warning("Quarterly EPS data unavailable.")

with col2:
    st.markdown("**Annual EPS (Last 4 Years):**")
    if not y_eps.empty:
        st.table(y_eps.tail(4)[['Earnings']])
    else:
        st.warning("Annual EPS data unavailable.")

# ----------- CSV Download Button ----------- 
st.download_button(
    label="⬇️ Download full dataset as CSV",
    data=df.to_csv().encode('utf-8'),
    file_name=f'{ticker}_stock_data.csv',
    mime='text/csv',
)

# ----------- Top Movers Section ----------- 
top_movers = [
    {"symbol": "GME", "price": 24.75, "percent_change": 3.75},
    {"symbol": "AAPL", "price": 157.85, "percent_change": -0.45},
    {"symbol": "AMZN", "price": 134.23, "percent_change": 2.65},
]

# Create a rolling ticker list for the top movers
st.markdown("""
    <div style="overflow-x:auto; white-space: nowrap; font-size: 20px;">
        <div class="top-movers">
            {}
        </div>
    </div>
    <style>
        .top-movers-item {
            display: inline-block;
            padding-right: 20px;
            font-weight: bold;
        }
        .up {
            color: green;
        }
        .down {
            color: red;
        }
    </style>
""".format(' '.join([f'<div class="top-movers-item"><span>{m["symbol"]}: <span class="{"up" if m["percent_change"] > 0 else "down"}">{m["price"]} ({m["percent_change"]:.2f}%)</span></span></div>' for m in top_movers])), unsafe_allow_html=True)
