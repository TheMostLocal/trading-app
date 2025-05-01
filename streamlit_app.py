import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

# Set page config
st.set_page_config(page_title="Stock Tracker", layout="wide")

# Title
st.title("📊 Stock Tracker - Last 3 Years")

# ----------- Load Data with Caching -----------
@st.cache_data(ttl=3600)
def load_price_data(ticker):
    df = yf.download(ticker, period="3y")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

@st.cache_data(ttl=3600)
def load_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    try:
        dta_ratio = round(stock.balance_sheet.loc["Total Liab"][0] / stock.balance_sheet.loc["Total Assets"][0], 2)
    except Exception:
        dta_ratio = "N/A"
    return {
        "EPS (TTM)": info.get("trailingEps", "N/A"),
        "Revenue (TTM)": f"${info.get('totalRevenue', 0):,}" if info.get("totalRevenue") else "N/A",
        "Debt-to-Assets Ratio": dta_ratio,
        "Market Cap": f"${info.get('marketCap', 0):,}" if info.get("marketCap") else "N/A",
        "PE Ratio": info.get("trailingPE", "N/A"),
        "Dividend Yield": f"{info.get('dividendYield', 'N/A') * 100 if info.get('dividendYield') else 'N/A'}%" 
    }

@st.cache_data(ttl=3600)
def load_eps_history(ticker):
    stock = yf.Ticker(ticker)
    try:
        q_eps = stock.quarterly_earnings
        y_eps = stock.earnings
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

# ----------- Top Movers Section ----------- 
# Placeholder for Top Movers (these would be dynamically fetched in a real app)
top_movers = [
    {"symbol": "GME", "price": 24.75, "percent_change": 3.75},
    {"symbol": "AAPL", "price": 157.85, "percent_change": -0.45},
    {"symbol": "AMZN", "price": 145.32, "percent_change": 1.22},
    {"symbol": "TSLA", "price": 307.14, "percent_change": -1.57}
]

st.markdown("""
    <style>
        .top-movers-header {
            font-size: 24px;
            font-weight: bold;
            color: #1E90FF;
            position: sticky;
            top: 0;
            background-color: #fff;
            z-index: 10;
            padding: 10px;
        }
        .top-movers-item {
            font-size: 18px;
            padding: 5px;
        }
        .up {color: green;}
        .down {color: red;}
    </style>
    <div class="top-movers-header">
        Top Movers:
    </div>
    <div class="top-movers-list">
        {}
    </div>
""".format(' '.join([f'<div class="top-movers-item"><span>{m["symbol"]}: <span class="{"up" if m["percent_change"] > 0 else "down"}">{m["price"]} ({m["percent_change"]:.2f}%)</span></span></div>' for m in top_movers])), unsafe_allow_html=True)

# ----------- Input Stock Ticker -----------
ticker_input = st.text_input("Enter Ticker Symbol:", value="GME").upper()

# ----------- Load Data -----------
df = load_price_data(ticker_input)
df = add_analytics(df)
financials = load_fundamentals(ticker_input)
q_eps, y_eps = load_eps_history(ticker_input)

# ----------- Historical Price Table -----------
st.subheader("📅 Historical Price Table (Last 3 Years)")
st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index(ascending=False))

# ----------- Average Price Calculations -----------
average_prices = {
    "Average Open": df['Open'].mean(),
    "Average High": df['High'].mean(),
    "Average Low": df['Low'].mean(),
    "Average Close": df['Close'].mean()
}

st.markdown("### Average Price Data:")
st.write(average_prices)

# ----------- Buy/Sell/Hold Signal ----------- 
# Just an example using a simple moving average crossover strategy (this could be more sophisticated)
def buy_sell_signal(df):
    if df['MA_5'].iloc[-1] > df['MA_25'].iloc[-1]:
        return "Buy"
    elif df['MA_5'].iloc[-1] < df['MA_25'].iloc[-1]:
        return "Sell"
    else:
        return "Hold"

signal = buy_sell_signal(df)
st.markdown(f"### Buy/Sell/Hold Signal: **{signal}**")

# ----------- Price Chart with Trend & MAs -----------
st.subheader("📈 Price Chart with Trend & MAs")

price_chart_data = df.reset_index()

base = alt.Chart(price_chart_data).encode(
    x='Date:T'
)

price_line = base.mark_line(color='white', strokeWidth=3).encode(
    y=alt.Y('Close:Q', scale=alt.Scale(domain=[0, price_chart_data['Close'].max() * 1.1]), title='Price')
)

ma_5 = base.mark_line(color='blue', strokeDash=[5, 3], size=3).encode(y='MA_5:Q')
ma_25 = base.mark_line(color='orange', strokeDash=[3, 3], size=3).encode(y='MA_25:Q')
ma_200 = base.mark_line(color='green', strokeDash=[1, 1], size=3).encode(y='MA_200:Q')
trend = base.mark_line(color='#FF9933', opacity=0.5).encode(y='Trend:Q')

st.altair_chart((price_line + ma_5 + ma_25 + ma_200 + trend).properties(height=400), use_container_width=True)

# ----------- Financials Summary ----------- 
st.subheader("💵 Key Financial Metrics")
financial_data = {
    "Market Cap": financials["Market Cap"],
    "EPS (TTM)": financials["EPS (TTM)"],
    "Revenue (TTM)": financials["Revenue (TTM)"],
    "PE Ratio": financials["PE Ratio"],
    "Dividend Yield": financials["Dividend Yield"],
    "Debt-to-Assets Ratio": financials["Debt-to-Assets Ratio"]
}
st.table(pd.DataFrame(financial_data.items(), columns=["Metric", "Value"]))

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
    file_name=f'{ticker_input}_stock_data.csv',
    mime='text/csv',
)
