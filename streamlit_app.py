import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("📊 Stock Tracker Dashboard")

# ----------- Ticker Selection -----------
ticker_symbol = st.text_input("Enter Stock Ticker (e.g., GME, AAPL):", "GME").upper()

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
        "Market Cap": f"${info.get('marketCap', 0):,}",
        "Trailing P/E": info.get("trailingPE", "N/A"),
        "Forward P/E": info.get("forwardPE", "N/A"),
        "PEG Ratio": info.get("pegRatio", "N/A"),
        "Price/Sales (TTM)": info.get("priceToSalesTrailing12Months", "N/A"),
        "Price/Book (MRQ)": info.get("priceToBook", "N/A"),
        "EPS (TTM)": info.get("trailingEps", "N/A"),
        "Revenue (TTM)": f"${info.get('totalRevenue', 0):,}" if info.get("totalRevenue") else "N/A",
        "EBITDA": f"${info.get('ebitda', 0):,}" if info.get("ebitda") else "N/A",
        "Return on Equity (ROE)": info.get("returnOnEquity", "N/A"),
        "Operating Margin": info.get("operatingMargins", "N/A"),
        "Implied Volatility": info.get("impliedVolatility", "N/A")
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

# ----------- Indicator Calculations -----------
def add_analytics(df):
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_25'] = df['Close'].rolling(window=25).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    df_reset = df.reset_index()
    df_reset['Date_ordinal'] = df_reset['Date'].map(pd.Timestamp.toordinal)
    coeffs = np.polyfit(df_reset['Date_ordinal'], df_reset['Close'], 1)
    df['Trend'] = coeffs[0] * df.index.map(pd.Timestamp.toordinal) + coeffs[1]

    # Buy/Hold/Sell Signal based on Moving Average Crossover
    latest_ma_5 = df['MA_5'].iloc[-1]
    latest_ma_25 = df['MA_25'].iloc[-1]
    if latest_ma_5 > latest_ma_25:
        df['Signal'] = 'Buy'
    elif latest_ma_5 < latest_ma_25:
        df['Signal'] = 'Sell'
    else:
        df['Signal'] = 'Hold'
    
    return df

# ----------- Load Data -----------
df = load_price_data(ticker_symbol)
df = add_analytics(df)
last_30 = df.tail(30)
financials = load_fundamentals(ticker_symbol)
q_eps, y_eps = load_eps_history(ticker_symbol)

# ----------- Display Rolling Ticker List at the Top ----------- 
# You can include tickers from top movers or top activity manually, or fetch from a service.
ticker_list = ['QQQ', 'SPY', 'NVDA','APPL','MSFT', 'TSLA', 'AMZN']  # Placeholder
ticker_data = []

# Fetch data for top movers, including today's gain/loss
for symbol in ticker_list:
    ticker_obj = yf.Ticker(symbol)
    data = ticker_obj.history(period="1d")
    price_change = data['Close'][0] - data['Open'][0]
    percent_change = (price_change / data['Open'][0]) * 100
    ticker_data.append({
        "symbol": symbol,
        "price_change": price_change,
        "percent_change": percent_change
    })

st.markdown("""
    <marquee style="font-size:20px;color:#FF6347;white-space:nowrap;">
    {} 
    </marquee>
    """.format(' '.join([f'<div class="top-movers-item"><span>{m["symbol"]}: <span class="{"up" if m["percent_change"] > 0 else "down"}">{m["price_change"]:+.2f} (${m["percent_change"]:+.2f}%)</span></span></div>' 
                        for m in ticker_data if isinstance(m["percent_change"], (float, int))])), unsafe_allow_html=True)

# ----------- Price Chart (3-Year) -----------
st.subheader("📈 Price Chart (3-Year)")

price_chart_data = df.reset_index()

line_chart = alt.Chart(price_chart_data).mark_line().encode(
    x='Date:T',
    y=alt.Y('Close:Q', title='Price'),
    color=alt.value('white'),
    tooltip=['Date:T', 'Close:Q', 'MA_5:Q', 'MA_25:Q', 'MA_200:Q']
).properties(height=400)

ma_5 = alt.Chart(price_chart_data).mark_line(color='blue', strokeDash=[4,2]).encode(
    x='Date:T', y='MA_5:Q'
)

ma_25 = alt.Chart(price_chart_data).mark_line(color='orange', strokeDash=[4,2]).encode(
    x='Date:T', y='MA_25:Q'
)

ma_200 = alt.Chart(price_chart_data).mark_line(color='green', strokeDash=[4,2]).encode(
    x='Date:T', y='MA_200:Q'
)

st.altair_chart((line_chart + ma_5 + ma_25 + ma_200).interactive(), use_container_width=True)

# ----------- Buy/Hold/Sell Signal (Under Price Chart) ----------- 
st.subheader(f"💡 {ticker_symbol} Buy/Hold/Sell Signal")
signal = df['Signal'].iloc[-1]  # Get the latest signal
st.markdown(f"**Signal:** {signal}")

# ----------- Volume Chart -----------
st.subheader("📊 Daily Volume (Last 30 Days)")

volume_chart_data = last_30.reset_index()
avg_volume = last_30['Volume'].mean()

volume_base = alt.Chart(volume_chart_data).encode(x='Date:T')

bars = volume_base.mark_bar(color="#4A90E2").encode(
    y=alt.Y('Volume:Q', title='Volume'),
    tooltip=['Date:T', 'Volume:Q']
)

avg_line = volume_base.mark_rule(color='red', strokeDash=[4, 2]).encode(
    y=alt.Y('Volume:Q')
).transform_calculate(
    Volume=str(avg_volume)
)

st.altair_chart((bars + avg_line).properties(height=200), use_container_width=True)
st.caption(f"🔻 Average Volume: {int(avg_volume):,}")

# ----------- Historical Price Table (Sorted) -----------
st.subheader("📅 Historical Price Table (Last 30 Days)")
st.dataframe(last_30[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index(ascending=False))

# ----------- Average OHLC Metrics -----------
st.subheader("📉 Average Price Metrics (Last 30 Days)")
st.markdown(f"""
- **Average Open:** ${last_30['Open'].mean():.2f}  
- **Average High:** ${last_30['High'].mean():.2f}  
- **Average Low:** ${last_30['Low'].mean():.2f}  
- **Average Close:** ${last_30['Close'].mean():.2f}
""")

# ----------- Financial Metrics Table -----------
st.subheader("💵 Key Financial Metrics")

metrics_df = pd.DataFrame.from_dict(financials, orient='index', columns=['Value'])
metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})
st.dataframe(metrics_df)

# ----------- EPS Display -----------
st.subheader("🧾 Earnings Per Share (EPS)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Last 8 Quarters EPS:**")
    if q_eps is not None and not q_eps.empty:
        st.table(q_eps.head(8)[['Earnings']])
    else:
        st.warning("Quarterly EPS data unavailable.")

with col2:
    st.markdown("**Annual EPS (Last 4 Years):**")
    if y_eps is not None and not y_eps.empty:
        st.table(y_eps.tail(4)[['Earnings']])
    else:
        st.warning("Annual EPS data unavailable.")

# ----------- CSV Download -----------
st.download_button(
    label="⬇️ Download full dataset as CSV",
    data=df.to_csv().encode('utf-8'),
    file_name=f'{ticker_symbol.lower()}_stock_data.csv',
    mime='text/csv',
)
