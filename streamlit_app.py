import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Stock Tracker", layout="wide")
st.title("📊 Stock Dashboard - Last 3 Years")

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
    financials = {
        "Market Cap": f"${info.get('marketCap', 0):,}" if info.get("marketCap") else "N/A",
        "Revenue (TTM)": f"${info.get('totalRevenue', 0):,}" if info.get("totalRevenue") else "N/A",
        "Gross Profit": f"${info.get('grossProfits', 0):,}" if info.get("grossProfits") else "N/A",
        "EBITDA": f"${info.get('ebitda', 0):,}" if info.get("ebitda") else "N/A",
        "Net Income": f"${info.get('netIncomeToCommon', 0):,}" if info.get("netIncomeToCommon") else "N/A",
        "EPS (TTM)": info.get("trailingEps", "N/A"),
        "P/E Ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A",
        "ROE": f"{round(info.get('returnOnEquity', 0)*100, 2)}%" if info.get("returnOnEquity") else "N/A",
        "ROA": f"{round(info.get('returnOnAssets', 0)*100, 2)}%" if info.get("returnOnAssets") else "N/A",
        "Profit Margin": f"{round(info.get('profitMargins', 0)*100, 2)}%" if info.get("profitMargins") else "N/A",
        "Current Ratio": round(info.get("currentRatio", 0), 2) if info.get("currentRatio") else "N/A",
        "Quick Ratio": round(info.get("quickRatio", 0), 2) if info.get("quickRatio") else "N/A",
        "Beta": round(info.get("beta", 0), 2) if info.get("beta") else "N/A"
    }
    return dict(sorted(financials.items(), key=lambda x: (x[1] == "N/A", x[0])))

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

# ----------- Ticker Input -----------
ticker_input = st.text_input("Enter a stock ticker (e.g., GME)", value="GME").upper()

# ----------- Load Data -----------
df = load_price_data(ticker_input)
df = add_analytics(df)
last_30 = df.tail(30)
financials = load_fundamentals(ticker_input)
q_eps, y_eps = load_eps_history(ticker_input)

# ----------- Price Table -----------
st.subheader("📅 Historical Price Table (Last 30 Days)")
st.dataframe(last_30[['Open', 'High', 'Low', 'Close', 'Volume']])

avg_open = last_30['Open'].mean()
avg_high = last_30['High'].mean()
avg_low = last_30['Low'].mean()
avg_close = last_30['Close'].mean()

st.markdown(f"**Average Open:** {avg_open:.2f} | **High:** {avg_high:.2f} | **Low:** {avg_low:.2f} | **Close:** {avg_close:.2f}")

# ----------- Altair Chart: Price + Trend + MAs -----------
st.subheader("📈 Price Chart with Trend & MAs")

price_chart_data = last_30.reset_index()

base = alt.Chart(price_chart_data).encode(x='Date:T')

hover = alt.selection_single(fields=['Date'], nearest=True, on='mouseover', empty='none', clear='mouseout')

lines = alt.layer(
    base.mark_line(color='white', strokeWidth=3).encode(y='Close:Q', tooltip=['Date:T', 'Close:Q']),
    base.mark_line(color='blue', strokeDash=[5, 3], size=2).encode(y='MA_5:Q'),
    base.mark_line(color='orange', strokeDash=[4, 3], size=2).encode(y='MA_25:Q'),
    base.mark_line(color='green', strokeDash=[2, 2], size=2).encode(y='MA_200:Q'),
    base.mark_line(color='red', opacity=0.5).encode(y='Trend:Q'),
    base.mark_rule(color='gray').encode(x='Date:T').add_selection(hover),
    base.mark_point().encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=['Date:T', 'Close:Q', 'MA_5:Q', 'MA_25:Q', 'MA_200:Q', 'Trend:Q']
    )
).resolve_scale(y='shared')

st.altair_chart(lines.properties(height=400), use_container_width=True)

# ----------- Daily Volume Chart -----------
st.subheader("📊 Daily Volume (Last 30 Days)")

avg_volume = last_30['Volume'].mean()
volume_chart = alt.Chart(price_chart_data).mark_bar(color="#4A90E2").encode(
    x='Date:T',
    y=alt.Y('Volume:Q', title='Volume'),
    tooltip=['Date:T', 'Volume:Q']
) + alt.Chart(price_chart_data).mark_rule(color="red", strokeDash=[4,2]).encode(
    y=alt.value(avg_volume)
)

st.altair_chart(volume_chart.properties(height=200), use_container_width=True)
st.caption(f"🔻 Average Volume: {int(avg_volume):,}")

# ----------- Financials Summary -----------
st.subheader("💵 Key Financial Metrics")
fin_df = pd.DataFrame(list(financials.items()), columns=["Metric", "Value"])
st.table(fin_df)

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

# ----------- Investment Recommendation -----------
st.subheader("📌 Investment Signal")
latest_price = df['Close'].iloc[-1]
ma_200 = df['MA_200'].iloc[-1]

if latest_price > ma_200:
    st.success("📈 Signal: BUY (Price above 200-day MA)")
elif latest_price < ma_200:
    st.error("📉 Signal: SELL (Price below 200-day MA)")
else:
    st.info("➖ Signal: HOLD")
