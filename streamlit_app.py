import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

# ----------- Sidebar Navigation ----------- 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Stock Dashboard", "Options & Implied Volatility"))

# ----------- Stock Dashboard ----------- 
if page == "Stock Dashboard":
    st.set_page_config(page_title="Stock Tracker Dashboard", layout="wide")
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
            "Operating Margin": info.get("operatingMargins", "N/A")
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
    ticker_list = ['GME', 'AAPL', 'MSFT', 'TSLA', 'AMZN']  # Placeholder
    st.markdown("""
        <marquee style="font-size:20px;color:#FF6347;white-space:nowrap;"> 
        {} 
        </marquee>
        """.format(', '.join(ticker_list)), unsafe_allow_html=True)

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

    # ----------- Historical Price Table ----------- 
    st.subheader("📅 Historical Price Table (Last 30 Days)")
    st.dataframe(last_30[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index(ascending=False))

    # ----------- Financial Metrics Table ----------- 
    st.subheader("💵 Key Financial Metrics")

    metrics_df = pd.DataFrame.from_dict(financials, orient='index', columns=['Value'])
    metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})
    st.dataframe(metrics_df)

    # ----------- Buy/Hold/Sell Signal ----------- 
    st.subheader(f"💡 {ticker_symbol} Buy/Hold/Sell Signal")
    signal = df['Signal'].iloc[-1]  # Get the latest signal
    st.markdown(f"**Signal:** {signal}")

# ----------- Options Page ----------- 
elif page == "Options & Implied Volatility":
    # Placeholder for your options data and implied volatility code
    st.title("📈 Options & Implied Volatility")
    st.markdown("Here you can see the options data and implied volatility for the selected stock.")
    st.write("Options and implied volatility data will be displayed here.")
    
    # Add the options and implied volatility functionality as described earlier
    # For example, call the functions that retrieve options and calculate implied volatility

