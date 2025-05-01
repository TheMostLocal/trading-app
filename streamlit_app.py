import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="GameStop (GME) Stock Tracker", layout="wide")
st.title("📊 GameStop (GME) Stock - Last 30 Days")

# ----------- Load Data with Caching -----------
@st.cache_data(ttl=3600)
def load_price_data():
    df = yf.download("GME", period="200d")
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
    return {
        "EPS (TTM)": info.get("trailingEps", "N/A"),
        "Revenue (TTM)": f"${info.get('totalRevenue', 0):,}" if info.get("totalRevenue") else "N/A",
        "Debt-to-Assets Ratio": dta_ratio
    }

@st.cache_data(ttl=3600)
def load_eps_history():
    ticker = yf.Ticker("GME")
    try:
        q_eps = ticker.quarterly_earnings
        y_eps = ticker.earnings
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

# ----------- Price Table -----------
st.subheader("📅 Historical Price Table (Last 30 Days)")
st.dataframe(last_30[['Open', 'High', 'Low', 'Close', 'Volume']])

# ----------- Altair Chart: Price + Trend + MAs -----------
st.subheader("📈 Price Chart with Trend & MAs")

price_chart_data = last_30.reset_index()

base = alt.Chart(price_chart_data).encode(
    x='Date:T'
)

price_line = base.mark_line(color='white', strokeWidth=3).encode(
    y=alt.Y('Close:Q', scale=alt.Scale(domain=[0, price_chart_data['Close'].max() * 1.1]), title='Price')
)

ma_5 = base.mark_line(color='#AAAAAA', strokeDash=[5, 3]).encode(y='MA_5:Q')
ma_25 = base.mark_line(color='#888888', strokeDash=[3, 3]).encode(y='MA_25:Q')
trend = base.mark_line(color='#FF9933', opacity=0.5).encode(y='Trend:Q')

st.altair_chart((price_line + ma_5 + ma_25 + trend).properties(height=400), use_container_width=True)

# ----------- Average Volume Chart -----------
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

# ----------- Financials Summary -----------
st.subheader("💵 Key Financial Metrics")
for k, v in financials.items():
    st.markdown(f"**{k}:** {v}")

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
    file_name='gme_stock_data.csv',
    mime='text/csv',
)
