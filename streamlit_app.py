import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="GameStop (GME) Stock Tracker", layout="wide")
st.title("📊 GameStop (GME) Stock Dashboard")

# ----------- Load Data with Caching -----------
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
        "Gross Profit (TTM)": f"${info.get('grossProfits', 0):,}" if info.get("grossProfits") else "N/A",
        "Operating Margin": f"{round(info.get('operatingMargins', 0) * 100, 2)}%" if info.get("operatingMargins") else "N/A",
        "Profit Margin": f"{round(info.get('profitMargins', 0) * 100, 2)}%" if info.get("profitMargins") else "N/A",
        "Return on Assets": f"{round(info.get('returnOnAssets', 0) * 100, 2)}%" if info.get("returnOnAssets") else "N/A",
        "Return on Equity": f"{round(info.get('returnOnEquity', 0) * 100, 2)}%" if info.get("returnOnEquity") else "N/A",
        "Book Value per Share": info.get("bookValue", "N/A"),
        "Beta": info.get("beta", "N/A")
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
df = load_price_data()
df = add_analytics(df)
last_30 = df.tail(30)
financials = load_fundamentals()
q_eps, y_eps = load_eps_history()

# ----------- Historical Price Table -----------
st.subheader("📅 Historical Price Table (Last 30 Days)")
st.dataframe(last_30[['Open', 'High', 'Low', 'Close', 'Volume']])

avg_open = last_30['Open'].mean()
avg_high = last_30['High'].mean()
avg_low = last_30['Low'].mean()
avg_close = last_30['Close'].mean()

st.markdown(f"**Average Open:** ${avg_open:.2f} | **High:** ${avg_high:.2f} | **Low:** ${avg_low:.2f} | **Close:** ${avg_close:.2f}")

# ----------- Price Chart -----------
st.subheader("📈 Price Chart with Trend & Moving Averages")
price_chart_data = last_30.reset_index()

base = alt.Chart(price_chart_data).encode(x='Date:T')
price_line = base.mark_line(color='white', strokeWidth=3).encode(y=alt.Y('Close:Q', title='Price'))
ma_5 = base.mark_line(color='#AAAAAA', strokeDash=[5, 3]).encode(y='MA_5:Q')
ma_25 = base.mark_line(color='#888888', strokeDash=[3, 3]).encode(y='MA_25:Q')
trend = base.mark_line(color='#FF9933', opacity=0.5).encode(y='Trend:Q')

st.altair_chart((price_line + ma_5 + ma_25 + trend).properties(height=400), use_container_width=True)

# ----------- Volume Chart -----------
st.subheader("📊 Daily Volume (Last 30 Days)")
avg_volume = last_30['Volume'].mean()
volume_chart = alt.Chart(price_chart_data).mark_bar(color="#4A90E2").encode(
    x='Date:T',
    y=alt.Y('Volume:Q', title='Volume')
) + alt.Chart(price_chart_data).mark_rule(color="red", strokeDash=[4,2]).encode(
    y=alt.value(avg_volume)
)
st.altair_chart(volume_chart.properties(height=200), use_container_width=True)
st.caption(f"🔻 Average Volume: {int(avg_volume):,}")

# ----------- Financial Metrics -----------
st.subheader("💵 Key Financial Metrics")
metrics_df = pd.DataFrame(financials.items(), columns=['Metric', 'Value'])
metrics_df = metrics_df.set_index('Metric')
st.table(metrics_df)

# ----------- EPS Section -----------
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

# ----------- Investment Indicator -----------
st.subheader("📌 Investment Indicator")
latest_price = df['Close'].iloc[-1]
ma_200 = df['MA_200'].iloc[-1]

if latest_price > ma_200:
    st.success(f"Current price (${latest_price:.2f}) is above the 200-day MA (${ma_200:.2f}): **Potential BUY signal**")
elif latest_price < ma_200:
    st.warning(f"Current price (${latest_price:.2f}) is below the 200-day MA (${ma_200:.2f}): **Potential SELL signal**")
else:
    st.info("Price is at the 200-day MA: **HOLD**")

# ----------- CSV Download Button -----------
st.download_button(
    label="⬇️ Download full dataset as CSV",
    data=df.to_csv().encode('utf-8'),
    file_name='gme_stock_data.csv',
    mime='text/csv',
)
