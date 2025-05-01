import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

# Set page config
st.set_page_config(page_title="Stock Tracker", layout="wide")
st.title("📊 Stock Tracker")

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
    return {
        "EPS (TTM)": info.get("trailingEps", "N/A"),
        "Revenue (TTM)": f"${info.get('totalRevenue', 0):,}" if info.get("totalRevenue") else "N/A",
        "PE Ratio": info.get("trailingPE", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "Beta": info.get("beta", "N/A"),
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

# ----------- Stock Picker ----------- 
ticker = st.text_input("Enter Stock Ticker:", value="GME", max_chars=5)

# ----------- Load Data -----------
df = load_price_data(ticker)
df = add_analytics(df)
last_30 = df.tail(30)
financials = load_fundamentals(ticker)
q_eps, y_eps = load_eps_history(ticker)

# ----------- Top Movers Section ----------- 
# Placeholder for Top Movers (these would be dynamically fetched in a real app)
top_movers = [
    {"symbol": "GME", "price": 24.75, "percent_change": 3.75},
    {"symbol": "AAPL", "price": 157.85, "percent_change": -0.45},
    {"symbol": "AMZN", "price": 145.32, "percent_change": 1.22},
    {"symbol": "TSLA", "price": 307.14, "percent_change": -1.57}
]

top_movers_html = ''.join([
    f'<div class="top-movers-item"><span>{m["symbol"]}: '
    f'<span class="{"up" if m["percent_change"] > 0 else "down"}">{m["price"]} ({m["percent_change"]:.2f}%)</span></span></div>'
    for m in top_movers
])

st.markdown(f"""
    <style>
        .top-movers-header {{
            font-size: 24px;
            font-weight: bold;
            color: #1E90FF;
            position: sticky;
            top: 0;
            background-color: #fff;
            z-index: 10;
            padding: 10px;
        }}
        .top-movers-item {{
            font-size: 18px;
            padding: 5px;
        }}
        .up {{color: green;}}
        .down {{color: red;}}
    </style>
    <div class="top-movers-header">
        Top Movers:
    </div>
    <div class="top-movers-list">
        {top_movers_html}
    </div>
""", unsafe_allow_html=True)

# ----------- Price Table (Reversed) -----------
st.subheader("📅 Historical Price Table (Last 30 Days)")
st.dataframe(last_30[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[::-1])

# ----------- Price Chart with Trend & MAs -----------
st.subheader("📈 Price Chart with Trend & MAs")

price_chart_data = last_30.reset_index()

base = alt.Chart(price_chart_data).encode(
    x='Date:T'
)

price_line = base.mark_line(color='white', strokeWidth=3).encode(
    y=alt.Y('Close:Q', scale=alt.Scale(domain=[0, price_chart_data['Close'].max() * 1.1]), title='Price')
)

ma_5 = base.mark_line(color='#5A9BD5', strokeDash=[5, 3]).encode(
    y='MA_5:Q', tooltip=['Date:T', 'Close:Q', 'MA_5:Q']
)
ma_25 = base.mark_line(color='#888888', strokeDash=[3, 3]).encode(
    y='MA_25:Q', tooltip=['Date:T', 'Close:Q', 'MA_25:Q']
)
ma_200 = base.mark_line(color='#FF9933', opacity=0.5).encode(
    y='MA_200:Q', tooltip=['Date:T', 'Close:Q', 'MA_200:Q']
)

st.altair_chart((price_line + ma_5 + ma_25 + ma_200).properties(height=400), use_container_width=True)

# ----------- Buy/Sell/Hold Signal ----------- 
# Placeholder for the signal logic (you can adjust this)
buy_sell_hold_signal = "Buy"  # Example placeholder value

st.subheader("📊 Buy/Sell/Hold Signal")
st.write(f"Signal: **{buy_sell_hold_signal}**")

# ----------- Daily Volume Chart ----------- 
st.subheader("📊 Daily Volume (Last 30 Days)")

avg_volume = last_30['Volume'].mean()
volume_chart = alt.Chart(price_chart_data).mark_bar(color="#4A90E2").encode(
    x='Date:T',
    y='Volume:Q'
)

st.altair_chart(volume_chart.properties(height=200), use_container_width=True)
st.caption(f"🔻 Average Volume: {int(avg_volume):,}")

# ----------- Financial Metrics ----------- 
st.subheader("💵 Key Financial Metrics")
financial_metrics = pd.DataFrame.from_dict(financials, orient='index', columns=["Value"]).sort_values(by="Value", ascending=False)
st.table(financial_metrics)

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
