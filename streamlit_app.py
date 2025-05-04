import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq

st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("📊 Stock Tracker Dashboard")

menu = st.selectbox("Select Page", ["Stock Dashboard", "Options & Implied Volatility", "Earnings Calendar"], key="menu")

# ---------- Common Functions ----------
@st.cache_data(ttl=3600)
def load_price_data(ticker, period="3y"):
    df = yf.download(ticker, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

@st.cache_data(ttl=3600)
def load_fundamentals(ticker):
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    def format_number(value):
        try:
            if abs(value) >= 1e9:
                return f"${value/1e9:.2f}B"
            elif abs(value) >= 1e6:
                return f"${value/1e6:.2f}M"
            elif abs(value) >= 1e3:
                return f"${value/1e3:.2f}K"
            else:
                return f"${value:.2f}"
        except:
            return "N/A"
    return {
        "Market Cap": f"${info.get('marketCap', 0):,}",
        "Beta": info.get("beta","N/A"),
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
    }

def add_analytics(df):
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_25'] = df['Close'].rolling(window=25).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_100'] = df['Close'].rolling(window=100).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']


    df_reset = df.reset_index()
    df_reset['Date_ordinal'] = df_reset['Date'].map(pd.Timestamp.toordinal)
    coeffs = np.polyfit(df_reset['Date_ordinal'], df_reset['Close'], 1)
    df['Trend'] = coeffs[0] * df.index.map(pd.Timestamp.toordinal) + coeffs[1]

    latest_ma_10 = df['MA_10'].iloc[-1]
    latest_ma_25 = df['MA_25'].iloc[-1]
    latest_ma_50 = df['MA_50'].iloc[-1]
    if latest_ma_10 > latest_ma_25:
        df['Signal'] = 'Buy'
    elif latest_ma_10 < latest_ma_25:
        df['Signal'] = 'Sell'
    elif latest_ma_25 > latest_ma_50:
        df['Signal'] = 'Hold'    
    else:
        df['Signal'] = 'Hold'

    return df
# ---------- Black-Scholes Functions ----------
def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2) / 100

    return delta, gamma, vega, theta, rho

def calculate_fibonacci_targets(df):
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    levels = [
        high,
        high - 0.236 * diff,
        high - 0.382 * diff,
        high - 0.5 * diff,
        high - 0.618 * diff,
        low
    ]
    return levels

# ---------- Stock Dashboard ----------
if menu == "Stock Dashboard":
    ticker_symbol = st.text_input("Enter Stock Ticker (e.g., GME, AAPL):", "GME").upper()

    timeframes = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "1 Year": "1y",
        "5 Years": "5y",
        "10 Years": "10y"
    }

    col1, col2 = st.columns([1, 4])
    with col1:
        selected_tf = st.selectbox("Timeframe", list(timeframes.keys()), index=2)

    df = load_price_data(ticker_symbol, timeframes[selected_tf])
    df = add_analytics(df)
    last_30 = df.tail(30)
    financials = load_fundamentals(ticker_symbol)
    if 'latest_iv' in st.session_state:
        financials['Implied Volatility (IV)'] = f"{st.session_state['latest_iv']:.2%}"
    st.subheader("\U0001F4B5 Key Financial Metrics")
    st.dataframe(pd.DataFrame.from_dict(financials, orient='index', columns=['Value']).reset_index().rename(columns={'index': 'Metric'}))

    tickers = ["QQQ", "SPY", "NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GME"]
    data = yf.download(tickers, period="1d", interval="1m")["Close"].ffill()
    latest = data.iloc[-1]
    previous = data.iloc[-2]

    # Build ticker HTML
    ticker_items = ""
    for ticker in tickers:
        current = latest[ticker]
        prev = previous[ticker]
        diff = current - prev
        pct = (diff / prev) * 100
        color = "#00FF00" if diff > 0 else "#FF4B4B"
    
        ticker_items += f"<span style='margin-right: 2rem; color: {color}; font-family: monospace; font-size: 16px;'>{ticker}: {current:.2f} ({diff:+.2f}, {pct:+.2f}%)</span>"


    # Scrolling ticker HTML with animation
    scroll_html = f"""
    <div style="width: 100%; overflow: hidden;">
        <div style="display: inline-block; white-space: nowrap; animation: ticker 20s linear infinite;">
            {ticker_items}
        </div>
    </div>

    <style>
    @keyframes ticker {{
        0% {{ transform: translateX(100%); }}
        100% {{ transform: translateX(-100%); }}
    }}
    </style>
    """

    st.markdown(scroll_html, unsafe_allow_html=True)

    st.subheader(f"📈 Price Chart ({selected_tf})")
    show_ma_10 = st.checkbox("Show 10-Day MA", value=False)
    show_ma_25 = st.checkbox("Show 25-Day MA", value=False)
    show_ma_50 = st.checkbox("Show 50-Day MA", value=False)
    show_ma_100 = st.checkbox("Show 100-Day MA", value=False)
    show_ma_200 = st.checkbox("Show 200-Day MA", value=False)
    show_fib = st.checkbox("Show Fibonacci Targets", value=False)
    show_bb = st.checkbox("Show Bollinger Bands", value=False)

    price_chart_data = df.reset_index()
    base_chart = alt.Chart(price_chart_data).mark_line().encode(
        x='Date:T',
        y=alt.Y('Close:Q', title='Price'),
        color=alt.value('white'),
        tooltip=['Date:T', 'Close:Q']
    ).properties(height=400)

    layers = [base_chart]
    if show_ma_10:
        layers.append(alt.Chart(price_chart_data).mark_line(color='blue', strokeDash=[4, 2]).encode(x='Date:T', y='MA_10:Q'))
    if show_ma_25:
        layers.append(alt.Chart(price_chart_data).mark_line(color='orange', strokeDash=[4, 2]).encode(x='Date:T', y='MA_25:Q'))
    if show_ma_50:
        layers.append(alt.Chart(price_chart_data).mark_line(color='purple', strokeDash=[4, 2]).encode(x='Date:T', y='MA_50:Q'))
    if show_ma_100:
        layers.append(alt.Chart(price_chart_data).mark_line(color='yellow', strokeDash=[4, 2]).encode(x='Date:T', y='MA_100:Q'))
    if show_ma_200:
        layers.append(alt.Chart(price_chart_data).mark_line(color='green', strokeDash=[4, 2]).encode(x='Date:T', y='MA_200:Q'))
    
    if show_fib:
        fib_levels = calculate_fibonacci_targets(df)
        for level in fib_levels:
            fib_line = alt.Chart(price_chart_data).mark_rule(color='gold', strokeDash=[2, 2]).encode(y=alt.datum(level))
            layers.append(fib_line)
    if show_bb:
        band = alt.Chart(price_chart_data).mark_area(opacity=0.2, color='lightblue').encode(
            x='Date:T',
            y='BB_Lower:Q',
            y2='BB_Upper:Q'
        )
        layers.append(band)

    st.altair_chart(alt.layer(*layers).interactive(), use_container_width=True)

    st.subheader(f"💡 {ticker_symbol} Buy/Hold/Sell Signal")
    signal = df['Signal'].iloc[-1]
    st.markdown(f"**Signal:** {signal}")

    st.subheader("📊 Daily Volume (Last 30 Days)")
    volume_chart_data = last_30.reset_index()
    avg_volume = last_30['Volume'].mean()
    volume_base = alt.Chart(volume_chart_data).encode(x='Date:T')
    bars = volume_base.mark_bar(color="#4A90E2").encode(y=alt.Y('Volume:Q'), tooltip=['Date:T', 'Volume:Q'])
    avg_line = volume_base.mark_rule(color='red', strokeDash=[4, 2]).encode(y=alt.Y('Volume:Q')).transform_calculate(Volume=str(avg_volume))
    st.altair_chart((bars + avg_line).properties(height=200), use_container_width=True)
    st.caption(f"🔻 Average Volume: {int(avg_volume):,}")

    st.subheader("📅 Historical Price Table (Last 30 Days)")
    st.dataframe(last_30[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index(ascending=False))

    st.subheader("📉 Average Price Metrics (Last 30 Days)")
    st.markdown(f"""
    - **Average Open:** ${last_30['Open'].mean():.2f}  
    - **Average High:** ${last_30['High'].mean():.2f}  
    - **Average Low:** ${last_30['Low'].mean():.2f}  
    - **Average Close:** ${last_30['Close'].mean():.2f}
    """)

# ---------- Options Page ----------
if menu == "Options & Implied Volatility":
    st.title("🛠️ Options & Implied Volatility")
    ticker_symbol = st.text_input("Enter Stock Ticker:", "AAPL").upper()
    ticker_obj = yf.Ticker(ticker_symbol)

    if not ticker_obj.options:
        st.warning("No options data available.")
    else:
        # Default to expiration closest to 30 days
        today = datetime.today()
        target_days = 30
        expiry_dates = [datetime.strptime(date, "%Y-%m-%d") for date in ticker_obj.options]
        expiry = min(expiry_dates, key=lambda x: abs((x - today).days - target_days)).strftime("%Y-%m-%d")

        expiry = st.selectbox("Select Expiration Date", ticker_obj.options, index=ticker_obj.options.index(expiry), key="expiry")
        option_type = st.radio("Option Type", ["call", "put"], horizontal=True)
        chain = ticker_obj.option_chain(expiry)
        options_df = chain.calls if option_type == "call" else chain.puts

        # Select nearest ATM
        spot_price = ticker_obj.history(period="1d")["Close"].iloc[-1]
        options_df["diff"] = np.abs(options_df["strike"] - spot_price)
        nearest = options_df.sort_values("diff").iloc[0]
        K = nearest["strike"]
        market_price = (nearest["bid"] + nearest["ask"]) / 2

        T = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days / 365
        r = 0.05

        def implied_volatility(S, K, T, r, market_price, option_type):
            from scipy.optimize import brentq
            def objective(sigma):
                d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                if option_type == "call":
                    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                else:
                    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                return price - market_price
            try:
                return brentq(objective, 0.0001, 3)
            except:
                return np.nan

        def black_scholes_greeks(S, K, T, r, sigma, option_type):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in volatility

            # Theta per day per contract (100 shares)
            if option_type == "call":
                theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            else:
                theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

            rho = (K * T * np.exp(-r * T) * norm.cdf(d2) / 100) if option_type == "call" else (-K * T * np.exp(-r * T) * norm.cdf(-d2) / 100)

            return delta, gamma, vega, theta, rho

        iv = implied_volatility(spot_price, K, T, r, market_price, option_type)
        delta, gamma, vega, theta, rho = black_scholes_greeks(spot_price, K, T, r, iv, option_type)

        st.markdown(f"**Selected Expiry:** {expiry}")
        st.markdown(f"**Strike Price (ATM):** {K}")
        st.markdown(f"**Option Premium (Mid Price):** ${market_price:.2f}")
        st.markdown(f"**Implied Volatility:** {iv:.2%}")

        st.subheader("\u03B3 Greeks")
        greeks_data = {
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega,
            "Theta (per day, per contract)": theta,
            "Rho": rho
        }
        st.dataframe(pd.DataFrame(greeks_data.items(), columns=["Greek", "Value"]))

        st.session_state['latest_iv'] = iv

# ---------- Earnings Calendar Page ----------
elif menu == "Earnings Calendar":
    st.title("📅 Earnings Calendar")
    earnings_calendar_data = pd.DataFrame({
        'Ticker': ['AAPL', 'GOOG', 'TSLA'],
        'Date': ['2025-05-01', '2025-05-02', '2025-05-03'],
        'Time': ['After Market', 'Before Market', 'After Market'],
        'Estimate': ['1.20', '2.50', '0.85']
    })
    st.write(earnings_calendar_data)
