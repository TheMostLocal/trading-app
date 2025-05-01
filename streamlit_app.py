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

# ----------- Price Chart with Trend & MAs -----------
st.subheader(f"📈 {ticker} Price Chart with Trend & MAs")

price_chart_data = df.reset_index()

base = alt.Chart(price_chart_data).encode(
    x='Date:T'
)

price_line = base.mark_line(color='white', strokeWidth=3).encode(
    y=alt.Y('Close:Q', scale=alt.Scale(domain=[0, price_chart_data['Close'].max() * 1.1]), title='Price')
)

ma_5 = base.mark_line(color='blue', strokeDash=[5, 3], size=3, tooltip=alt.Tooltip(field="MA_5", type="quantitative", title="5-day MA")).encode(y='MA_5:Q')
ma_25 = base.mark_line(color='green', strokeDash=[5, 3], size=3, tooltip=alt.Tooltip(field="MA_25", type="quantitative", title="25-day MA")).encode(y='MA_25:Q')
ma_200 = base.mark_line(color='red', strokeDash=[5, 3], size=3, tooltip=alt.Tooltip(field="MA_200", type="quantitative", title="200-day MA")).encode(y='MA_200:Q')
trend = base.mark_line(color='#FF9933', opacity=0.5, size=3, tooltip=alt.Tooltip(field="Trend", type="quantitative", title="Trend")).encode(y='Trend:Q')

st.altair_chart((price_line + ma_5 + ma_25 + ma_200 + trend).properties(height=400), use_container_width=True)

# ----------- Sentiment ----------- 
# Placeholder for Sentiment (could be based on fundamental analysis or technical indicators)
sentiment = "Buy"  # In real scenario, it could be computed using some indicators
st.markdown(f"**Sentiment:** {sentiment}", unsafe_allow_html=True)

# ----------- Historical Price Data (Latest at top) -----------
st.subheader("📅 Historical Price Data (Last 30 Days)")

last_30_sorted = last_30[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[::-1]  # Latest at top

# Add Average Price Stats
avg_prices = last_30[['Open', 'High', 'Low', 'Close']].mean()
st.markdown(f"**Average Prices:** Open: ${avg_prices['Open']:.2f}, High: ${avg_prices['High']:.2f}, Low: ${avg_prices['Low']:.2f}, Close: ${avg_prices['Close']:.2f}")

st.dataframe(last_30_sorted)

# ----------- Financial Metrics Table ----------- 
st.subheader("📊 Financial Metrics")
financial_metrics = pd.DataFrame.from_dict(financials, orient='index', columns=["Value"])
st.dataframe(financial_metrics)

# ----------- Earnings Per Share (EPS) ----------- 
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

# ----------- Daily Volume Chart ----------- 
st.subheader("📊 Daily Volume (Last 30 Days)")

volume_chart = alt.Chart(price_chart_data).mark_bar(color="#4A90E2").encode(
    x='Date:T',
    y='Volume:Q'
)

st.altair_chart(volume_chart.properties(height=200), use_container_width=True)
st.caption(f"🔻 Average Volume: {int(last_30['Volume'].mean()):,}")

# ----------- CSV Download Button -----------
st.download_button(
    label="⬇️ Download full dataset as CSV",
    data=df.to_csv().encode('utf-8'),
    file_name=f"{ticker}_stock_data.csv",
    mime="text/csv"
)
