import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Stock Tracker", layout="wide")
st.title("📊 Stock Tracker - Last 3 Years")

# ----------- Ticker Input -----------
ticker_input = st.text_input("Enter a stock ticker (e.g., GME, AAPL, TSLA):", value="GME").upper()

# ----------- Load Data -----------
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
        "Gross Profit Margin": f"{(info.get('grossMargins', 0) * 100):.2f}%" if info.get('grossMargins') else "N/A",
        "Operating Margin": f"{(info.get('operatingMargins', 0) * 100):.2f}%" if info.get('operatingMargins') else "N/A",
        "Net Profit Margin": f"{(info.get('profitMargins', 0) * 100):.2f}%" if info.get('profitMargins') else "N/A",
        "Return on Assets (ROA)": f"{(info.get('returnOnAssets', 0) * 100):.2f}%" if info.get('returnOnAssets') else "N/A",
        "Return on Equity (ROE)": f"{(info.get('returnOnEquity', 0) * 100):.2f}%" if info.get('returnOnEquity') else "N/A",
        "P/E Ratio": f"{info.get('trailingPE', 'N/A')}",
        "Price-to-Sales (P/S) Ratio": f"{info.get('priceToSalesTrailing12Months', 'N/A')}",
        "Enterprise Value": f"${info.get('enterpriseValue', 0):,}" if info.get('enterpriseValue') else "N/A",
        "EV/EBITDA": f"{info.get('enterpriseToEbitda', 'N/A')}",
        "Current Ratio": f"{info.get('currentRatio', 'N/A')}",
        "Quick Ratio": f"{info.get('quickRatio', 'N/A')}",
        "Interest Coverage Ratio": f"{info.get('ebitda', 'N/A')} / {info.get('interestExpense', 'N/A')}",
        "Asset Turnover Ratio": f"{info.get('assetTurnover', 'N/A')}",
        "Inventory Turnover": f"{info.get('inventoryTurnover', 'N/A')}",
        "Market Cap": f"${info.get('marketCap', 0):,}" if info.get('marketCap') else "N/A",
        "Beta": f"{info.get('beta', 'N/A')}",
        "52-Week High": f"${info.get('fiftyTwoWeekHigh', 'N/A')}",
        "52-Week Low": f"${info.get('fiftyTwoWeekLow', 'N/A')}",
        "Shares Outstanding": f"{info.get('sharesOutstanding', 'N/A'):,}" if info.get('sharesOutstanding') else "N/A",
        "Dividend Yield": f"{(info.get('dividendYield', 0) * 100):.2f}%" if info.get('dividendYield') else "N/A",
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

# Load all data
df = load_price_data(ticker_input)
df = add_analytics(df)
last_30 = df.tail(30)
financials = load_fundamentals(ticker_input)
q_eps, y_eps = load_eps_history(ticker_input)

# ----------- Historical Price Table -----------
st.subheader("📅 Historical Price Table (Last 30 Days)")
st.dataframe(last_30[['Open', 'High', 'Low', 'Close', 'Volume']])

# ----------- Price + MAs Chart -----------
st.subheader(f"📈 Price Chart with Trend & MAs - {ticker_input}")
chart_data = df.reset_index()[['Date', 'Close', 'MA_5', 'MA_25', 'MA_200', 'Trend']]
melted = chart_data.melt(id_vars='Date', var_name='Indicator', value_name='Value')

line_chart = alt.Chart(melted).mark_line(size=2).encode(
    x=alt.X('Date:T'),
    y=alt.Y('Value:Q', title='Price / Moving Averages'),
    color=alt.Color('Indicator:N', legend=alt.Legend(title="Legend")),
    tooltip=[
        alt.Tooltip('Date:T', title='Date'),
        alt.Tooltip('Indicator:N', title='Metric'),
        alt.Tooltip('Value:Q', title='Value', format=".2f")
    ]
).properties(height=400).interactive()

st.altair_chart(line_chart, use_container_width=True)

# ----------- Volume Chart -----------
st.subheader(f"📊 Daily Volume (Last 30 Days) - {ticker_input}")
volume_data = last_30.reset_index()[['Date', 'Volume']]
avg_volume = volume_data['Volume'].mean()

volume_chart = alt.Chart(volume_data).mark_bar(color="#4A90E2", opacity=0.6).encode(
    x=alt.X('Date:T', title='Date'),
    y=alt.Y('Volume:Q', title='Volume'),
    tooltip=[
        alt.Tooltip('Date:T', title='Date'),
        alt.Tooltip('Volume:Q', format=',', title='Volume')
    ]
)

avg_line = alt.Chart(volume_data).mark_rule(color='red', strokeDash=[4, 2]).encode(
    y='mean(Volume):Q'
)

st.altair_chart((volume_chart + avg_line).properties(height=200), use_container_width=True)
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
    file_name=f'{ticker_input}_stock_data.csv',
    mime='text/csv'
)
