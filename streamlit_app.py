import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="GME Stock Dashboard", layout="wide")
st.title("📊 GameStop (GME) Stock Tracker")

# ----------- Load Data -----------
@st.cache_data(ttl=3600)
def load_price_data():
    df = yf.download("GME", period="3y")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

@st.cache_data(ttl=3600)
def load_fundamentals():
    ticker = yf.Ticker("GME")
    info = ticker.info
    return {
        "EPS (TTM)": info.get("trailingEps", "N/A"),
        "Revenue (TTM)": f"${info.get('totalRevenue', 0):,}" if info.get("totalRevenue") else "N/A",
        "Net Income (TTM)": f"${info.get('netIncomeToCommon', 0):,}" if info.get("netIncomeToCommon") else "N/A",
        "Gross Margin (%)": f"{info.get('grossMargins', 0)*100:.2f}%" if info.get("grossMargins") else "N/A",
        "Operating Margin (%)": f"{info.get('operatingMargins', 0)*100:.2f}%" if info.get("operatingMargins") else "N/A",
        "Return on Assets (ROA)": f"{info.get('returnOnAssets', 0)*100:.2f}%" if info.get("returnOnAssets") else "N/A",
        "Return on Equity (ROE)": f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get("returnOnEquity") else "N/A",
        "P/E Ratio": info.get("trailingPE", "N/A"),
        "Price-to-Book (P/B) Ratio": info.get("priceToBook", "N/A"),
        "Current Ratio": info.get("currentRatio", "N/A"),
        "Quick Ratio": info.get("quickRatio", "N/A")
    }

@st.cache_data(ttl=3600)
def load_eps_history():
    ticker = yf.Ticker("GME")
    try:
        q_eps = ticker.quarterly_earnings or pd.DataFrame()
        y_eps = ticker.earnings or pd.DataFrame()
    except Exception:
        q_eps = pd.DataFrame()
        y_eps = pd.DataFrame()
    return q_eps, y_eps

# ----------- Add Technical Indicators -----------
def add_analytics(df):
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_25'] = df['Close'].rolling(window=25).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['Signal_Line'] = compute_macd(df['Close'])
    df['Trend'] = np.poly1d(np.polyfit(df.index.map(pd.Timestamp.toordinal), df['Close'], 1))(df.index.map(pd.Timestamp.toordinal))
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# ----------- Load Data -----------
df = load_price_data()
df = add_analytics(df)
last_30 = df.tail(30)
financials = load_fundamentals()
q_eps, y_eps = load_eps_history()

# ----------- Price Table -----------
st.subheader("📅 Historical Price Table (Last 30 Days)")
st.dataframe(last_30[['Open', 'High', 'Low', 'Close', 'Volume']])

avg_open = last_30['Open'].mean()
avg_high = last_30['High'].mean()
avg_low = last_30['Low'].mean()
avg_close = last_30['Close'].mean()
st.markdown(f"**📈 Average Prices (Last 30 Days):**  ")
st.markdown(f"- **Open:** ${avg_open:.2f}  ")
st.markdown(f"- **High:** ${avg_high:.2f}  ")
st.markdown(f"- **Low:** ${avg_low:.2f}  ")
st.markdown(f"- **Close:** ${avg_close:.2f}")

# ----------- Altair Chart -----------
st.subheader("📈 3-Year Price Chart with Trend & MAs")
price_chart_data = df.reset_index()

chart = alt.Chart(price_chart_data).transform_fold(
    ['Close', 'MA_5', 'MA_25', 'MA_200']
).mark_line().encode(
    x='Date:T',
    y='value:Q',
    color=alt.Color('key:N', title='Legend'),
    tooltip=[alt.Tooltip('Date:T'),
             alt.Tooltip('Close:Q', title='Price'),
             alt.Tooltip('MA_5:Q', title='5-day MA'),
             alt.Tooltip('MA_25:Q', title='25-day MA'),
             alt.Tooltip('MA_200:Q', title='200-day MA')]
).properties(height=400)

st.altair_chart(chart, use_container_width=True)

# ----------- Volume Chart -----------
st.subheader("📊 Daily Volume (Last 30 Days)")
volume_chart_data = last_30.reset_index()
avg_volume = last_30['Volume'].mean()

volume_chart = alt.Chart(volume_chart_data).mark_bar(color="#4A90E2").encode(
    x='Date:T',
    y=alt.Y('Volume:Q', title='Volume'),
    tooltip=['Date:T', 'Volume:Q']
) + alt.Chart(volume_chart_data).mark_rule(color="red", strokeDash=[4,2]).encode(
    y=alt.Value(avg_volume)
)

st.altair_chart(volume_chart.properties(height=200), use_container_width=True)
st.caption(f"🔻 Average Volume: {int(avg_volume):,}")

# ----------- Financials Summary -----------
st.subheader("💵 Key Financial Metrics")
financial_order = [
    "Quick Ratio", "Current Ratio", "Price-to-Book (P/B) Ratio", "P/E Ratio", 
    "Return on Equity (ROE)", "Return on Assets (ROA)", "Operating Margin (%)", 
    "Gross Margin (%)", "Net Income (TTM)", "Revenue (TTM)", "EPS (TTM)"
]

sorted_financials = {k: financials.get(k, 'N/A') for k in financial_order}
metrics_df = pd.DataFrame(sorted_financials.items(), columns=["Metric", "Value"])
st.table(metrics_df)

# ----------- EPS Section -----------
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

# ----------- Signal -----------
st.subheader("💡 Investment Signal")
latest = df.iloc[-1]
if latest['Close'] > latest['MA_200']:
    st.success("📈 Price is above 200-day MA → Potential **Buy** Signal")
elif latest['Close'] < latest['MA_200']:
    st.error("📉 Price is below 200-day MA → Potential **Sell** Signal")
else:
    st.info("⏸️ Price is near 200-day MA → Consider **Hold**")

# ----------- CSV Download -----------
st.download_button(
    label="⬇️ Download full dataset as CSV",
    data=df.to_csv().encode('utf-8'),
    file_name='gme_stock_data.csv',
    mime='text/csv',
)
